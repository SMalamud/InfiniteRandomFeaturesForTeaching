from typing import Optional
import sys

from helpers.auxilliary_functions import simple_markowitz

sys.path.append('../infinite_features_for_AQR')
import numpy as np
import pandas as pd
import time
from data_preprocessing.process_futures_data import Futures
from rf.RandomFeaturesGeneratorr import RandomFeaturesGeneratorr


def smooth_positions_naively(returns: pd.DataFrame,
                             transaction_costs: dict,
                             positions: pd.DataFrame,
                             scales: list,
                             shrinkage_list: list,
                             rolling_window: int,
                             starting_point: int):
    """
    Careful, positions are supposed to be aligned
    :param returns:
    :param transaction_costs:
    :param positions:
    :param scales:
    :param shrinkage_list:
    :param rolling_window:
    :param starting_point:
    :return:
    """
    rets = returns.values
    pos = positions.values
    tickers = returns.columns
    lam = np.diag([transaction_costs[ticker] for ticker in tickers])
    smoothed_positions = {f'shr_{z}_scale_{scale}': pos.copy() for z in shrinkage_list for scale in scales}

    for i_ in range(starting_point, returns.shape[0]):
        rr = rets[(i_ - rolling_window):i_, :]
        sigma = rr.T @ rr
        identity = np.eye(sigma.shape[0])
        for scale in scales:
            for z_ in shrinkage_list:
                matr = np.linalg.inv(sigma + z_ * identity + scale * lam) * scale * np.diag(lam).reshape(1, -1)
                # here we are being careful, multiplying old positions by returns to account for drift
                smoothed_positions[f'shr_{z_}_scale_{scale}'][i_, :] \
                    = (matr @ ((smoothed_positions[f'shr_{z_}_scale_{scale}'][i_ - 1, :] * (1 + rets[i_, :])).reshape(-1, 1))
                       + (identity - matr) @ (pos[i_, :].reshape(-1, 1))).flatten()
    for scale in scales:
        for z_ in shrinkage_list:
            smoothed_positions[f'shr_{z_}_scale_{scale}'] = pd.DataFrame(smoothed_positions[f'shr_{z_}_scale_{scale}'],
                                                                         index=positions.index,
                                                                         columns=positions.columns)
    return smoothed_positions


def position_change(positions: pd.DataFrame,
                    returns: pd.DataFrame,
                    account_for_position_drift: bool = True):
    """
    If we account for position drift then we need to multiply positions with returns
    :param positions:
    :param returns:
    :param account_for_position_drift:
    :return:
    """
    if account_for_position_drift:
        return positions - (positions.shift(1) * (1 + returns))
    else:
        return positions.diff()


def compute_transaction_costs_for_msrr(signals_dictionary_indexed_by_ticker: dict,
                                       transaction_costs: dict,
                                       returns: pd.DataFrame,
                                       account_for_position_drift: bool = True,
                                       return_summed_cost: bool = True
                                       ):
    """
    We are trading \sum_k \sum_j \pi_j(k) S_{j,t}(k) R_{j,t+1}
    and the t-costs are
    \sum_t \sum_j \Lambda(j) (\sum_k \pi_j(k) \Delta S_{j,t}(k))^2
    = \sum_j \Lambda(j)\sum_{k_1,k_2} \pi_j(k_1)\pi_j(k_2) \Delta S_{j,t}(k_1)\Delta S_{j,t}(k_2)

    So, if we do factors, then we need to sum those matrices across j.
    And if we do MSRR, then we need to keep all of them
    :param signals_dictionary:
    :param transaction_costs:
    :param account_for_position_drift:
    :return: tuple of (summed t-cost, dict of costs) of dimension "number_factors"
    """
    tickers = returns.columns  # this is important: returns order is fixed now

    num_factors = signals_dictionary_indexed_by_ticker[tickers[0]].shape[1]
    num_periods = signals_dictionary_indexed_by_ticker[tickers[0]].shape[0]

    effective_t_cost_matrix = np.zeros([num_factors, num_factors])

    effective_t_cost_dict = dict()

    for ticker in tickers:
        # now we loop through tickers in the order of returns
        pos_change = position_change(signals_dictionary_indexed_by_ticker[ticker],
                                     returns[ticker].values.reshape(-1, 1),
                                     account_for_position_drift=account_for_position_drift)
        pos_change = np.nan_to_num(pos_change, posinf=0, neginf=0)
        tmp = pos_change.T @ pos_change
        effective_t_cost_dict[ticker] = tmp * transaction_costs[ticker]

        if return_summed_cost:
            try:
                effective_t_cost_matrix += effective_t_cost_dict[ticker]
            except:
                breakpoint()
    effective_t_cost_matrix /= num_periods
    return effective_t_cost_matrix, effective_t_cost_dict


def build_mssr_matrix_with_t_costs(msrr_returns: np.ndarray,
                                   effective_t_cost_dict: dict,
                                   effective_t_cost_matr: np.ndarray,
                                   ticker_list: list,
                                   scale: float = 1.,
                                   factors: bool = False):
    """
    We return CovarianceMatrix(msrr_returns) + scale *  effective_t_cost_matr
    if factors=True

    If factors=False, then we are dealing with msrr, and hence we need to do it per ticker
    and

    :param msrr_returns:
    :param effective_t_cost_dict:
    :param effective_t_cost_matr:
    :param scale: we are maximizing scale + Covariance + Tcost. Thus, larger scale means we care less about t-cost
    :param factors:
    :return:
    """
    tmp = msrr_returns.T @ msrr_returns / msrr_returns.shape[0]
    if not factors:
        count = 0
        for ticker in ticker_list:
            # it is important that the ticker list is correctly ordered
            # because we are asusming that msrr_returns are coming in the order of tickers
            # when we define count:count_next
            count_next = count + effective_t_cost_dict[ticker].shape[1]
            tmp[count:count_next, count:count_next] = tmp[count:count_next, count:count_next] \
                                                      + scale * effective_t_cost_dict[
                                                          ticker]
    else:
        tmp += scale * effective_t_cost_matr
    return tmp


def mssr_optimal_portfolio_with_costs(msrr_returns: np.ndarray,
                                      effective_t_cost_dict: dict,
                                      effective_t_cost: np.ndarray,
                                      scale: float,
                                      shrinkage_list: list,
                                      ticker_list: list,
                                      factors: bool = False):
    """
    B = UDU'
    We use that (z+B)^{-1}mu = U (D+z)^{-1}U'mu
    :param factors: whether we are doing factors or MSRR
    :param effective_t_cost: this is the matrix of t-costs averaged across tickers
    :param msrr_returns: pre-computed managed returns
    :param effective_t_cost_dict: dictionary of effective costs matrices per ticker
    :param scale: we are maximizing scale + Covariance + Tcost. Thus, larger scale means we care less about t-costs
    :param shrinkage_list:
    :return:
    """
    matr = build_mssr_matrix_with_t_costs(msrr_returns,
                                          effective_t_cost_dict,
                                          effective_t_cost,
                                          ticker_list,
                                          scale,
                                          factors)

    tmp = simple_markowitz(matr,
                           msrr_returns,
                           shrinkage_list)
    return tmp


def generate_returns_mssr_and_factors(slice_: dict,
                                      test_or_train: str,
                                      pre_specified_list_of_specs_for_random_features: list = None,
                                      number_features_per_ticker: int = None,
                                      seed: int = 0,
                                      seed_step: int = 1000,
                                      just_return_original_features: bool = True):
    random_features_all = dict()
    managed_returns_all = []
    tickers = slice_[f'{test_or_train}_ret'].columns  # fix the tickers! Careful, the order is fixed in pandas

    for ii, ticker in enumerate(tickers):
        random_features = (
            RandomFeaturesGeneratorr.generate_random_features_from_lisst(
                seed=int((seed + 1) * seed_step),
                features=slice_[f'{test_or_train}_sigs'][ticker].values,
                pre_specified_list_of_specs=pre_specified_list_of_specs_for_random_features,
                number_features_in_subset=number_features_per_ticker,
                just_return_original_features=just_return_original_features
            )
        )
        managed_returns_all += [slice_[f'{test_or_train}_ret'].values[:, ii].reshape(-1, 1) * random_features]
        random_features_all[ticker] = random_features

    msrr_returns = np.concatenate(managed_returns_all, axis=1)

    factor_returns = np.zeros(managed_returns_all[0].shape)
    for i in range(len(managed_returns_all)):
        factor_returns += managed_returns_all[i]
    in_sample_stuff = {'features': random_features_all, 'msrr_rets': msrr_returns, 'factor_rets': factor_returns}
    return in_sample_stuff


def reorganize_positions(oos_factor_positions):
    """
    positions come as a list indexed by ticker number; we want to reshuffle them, so that they become
    a list indexed by shrinkage
    :param oos_factor_positions:
    :return:
    """
    num_tickers = len(oos_factor_positions)
    number_versions = oos_factor_positions[0].shape[1]
    oos_factor_positions = [
        np.concatenate([oos_factor_positions[i][:, j].reshape(-1, 1) for i in range(num_tickers)], axis=1)
        for j in range(number_versions)]
    return oos_factor_positions


def giant_msrr_with_t_cost(seed: int,
                           number_features_per_ticker: int,
                           slice_: dict,
                           shrinkage_list: list,
                           scales_list: list,
                           tcosts: dict,
                           pre_specified_list_of_specs_for_random_features: list = None,
                           seed_step: int = 1e3,
                           account_for_position_drift: bool = True,
                           use_original_signals: bool = True,
                           test: bool = False):
    # first we start by fixing the tickers order. This is important, otherwise dictionary makes this random
    tickers = list(slice_['train_ret'].columns)

    t1 = time.time()
    effective_t_cost_matrix, effective_t_cost_dict = \
        compute_transaction_costs_for_msrr(signals_dictionary_indexed_by_ticker=slice_['train_sigs'],
                                           transaction_costs=tcosts,
                                           returns=slice_['train_ret'],
                                           account_for_position_drift=account_for_position_drift,
                                           return_summed_cost=True
                                           )
    t2 = time.time()
    print(f'computing tcosts took {t2 - t1}')

    in_sample_stuff = generate_returns_mssr_and_factors(slice_,
                                                        'train',
                                                        pre_specified_list_of_specs_for_random_features,
                                                        number_features_per_ticker,
                                                        seed,
                                                        seed_step,
                                                        just_return_original_features=use_original_signals)
    t3 = time.time()
    print(f'getting returns took {t3 - t2}')

    portfolios_msrr = np.concatenate([mssr_optimal_portfolio_with_costs(in_sample_stuff['msrr_rets'],
                                                                        effective_t_cost_dict=effective_t_cost_dict,
                                                                        effective_t_cost=effective_t_cost_matrix,
                                                                        scale=scale,
                                                                        shrinkage_list=shrinkage_list,
                                                                        ticker_list=tickers,
                                                                        factors=False)
                                      for scale in scales_list], axis=1)

    t4 = time.time()
    print(f'computing msrr portfolio took {t4 - t3}')

    portfolios_factors = np.concatenate([mssr_optimal_portfolio_with_costs(in_sample_stuff['factor_rets'],
                                                                           effective_t_cost_dict=effective_t_cost_dict,
                                                                           effective_t_cost=effective_t_cost_matrix,
                                                                           scale=scale,
                                                                           ticker_list=tickers,
                                                                           shrinkage_list=shrinkage_list,
                                                                           factors=True)
                                         for scale in scales_list], axis=1)

    t5 = time.time()
    print(f'computing factor portfolio took {t5 - t4}, factors shape={in_sample_stuff["factor_rets"].shape}')

    oos_stuff = generate_returns_mssr_and_factors(slice_,
                                                  'test',
                                                  pre_specified_list_of_specs_for_random_features,
                                                  number_features_per_ticker,
                                                  seed,
                                                  seed_step,
                                                  just_return_original_features=use_original_signals)

    t6 = time.time()

    print(f'oos return generation took {t6 - t5}')

    oos_msrr_retuns = oos_stuff['msrr_rets'] @ portfolios_msrr
    oos_factor_retuns = oos_stuff['factor_rets'] @ portfolios_factors

    num_signals = portfolios_factors.shape[0]
    # concatenated = np.concatenate(oos_msrr_retuns['features'], axis=1)

    # now for each i this is an array of dimensions T_{OOS} \times number_versions
    oos_msrr_positions = [oos_stuff['features'][ticker] @ portfolios_msrr[(i * num_signals):((i + 1) * num_signals),
                                                          :] for i, ticker in enumerate(tickers)]
    oos_msrr_positions = reorganize_positions(oos_msrr_positions)

    oos_factor_positions = [oos_stuff['features'][ticker] @ portfolios_factors for i, ticker in enumerate(tickers)]
    oos_factor_positions = reorganize_positions(oos_factor_positions)

    if test:
        number_versions = len(oos_factor_positions)
        tested_factor_rets = np.concatenate(
            [(oos_factor_positions[j] * slice_[f'test_ret'].values).sum(1).reshape(-1, 1)
             for j in range(number_versions)], axis=1)
        if np.abs(tested_factor_rets - oos_factor_retuns).mean() > 0.001:
            breakpoint()

        tested_msrr_rets = np.concatenate([(oos_msrr_positions[j] * slice_[f'test_ret'].values).sum(1).reshape(-1, 1)
                                           for j in range(number_versions)], axis=1)

        if np.abs(tested_msrr_rets - oos_msrr_retuns).mean() > 0.001:
            breakpoint()

    cols = [f'sc_{scale}_shr_{z}' for scale in scales_list for z in shrinkage_list]

    oos_msrr_retuns = pd.DataFrame(oos_msrr_retuns,
                                   index=slice_[f'test_ret'].index,
                                   columns=[f'msrr_{x}' for x in cols])

    oos_factor_retuns = pd.DataFrame(oos_factor_retuns,
                                     index=slice_[f'test_ret'].index,
                                     columns=[f'factor_{x}' for x in cols])

    strategy_returns = pd.concat([oos_msrr_retuns, oos_factor_retuns], axis=1)
    t7 = time.time()
    print(f'final steps took {t7 - t6}')
    return {'strategy_returns': strategy_returns,
            'msrr_positions': oos_msrr_positions,
            'factor_positions': oos_factor_positions}
