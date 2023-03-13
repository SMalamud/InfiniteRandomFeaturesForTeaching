import itertools

import scipy.stats.mstats as mstats
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time


def smart_ranking(data: np.ndarray,
                  axis: int,
                  ranking: str):
    """
    The function replaces data with their ranks, normalized to be in [-0.5, 0.5]
    Parameters
    ----------
    data :
    axis :
    ranking :

    Returns
    -------

    """
    indic = ~np.isnan(data)
    enough = indic.sum(axis) >= 4  # at least four instruments
    if axis == 0:
        indic = indic * enough.reshape(1, -1)
    else:
        indic = indic * enough.reshape(-1, 1)
    if ranking == 'rank':
        ranked_data = ((mstats.rankdata(np.ma.masked_invalid(data), axis=axis)) / (~np.isnan(data)).sum(
            axis) - 0.5) * indic
        # ranked_data = ((data.argsort(axis=axis).argsort(axis=axis) + 1) / data.shape[axis] - 0.5) * indic
        return ranked_data  # semyon
        # return (rankdata(data, axis=axis, method='average')) / (data.shape[axis]) + 0.5 # correct
    elif ranking == 'cdf':
        mu = np.nanmean(data, axis)
        sigma = np.nanstd(data, axis)
        return np.nan_to_num(semyons_erf((data - mu) / sigma) * 0.5) * indic


def semyons_erf(x: np.ndarray):
    """
    This is Semyon's implementation of erf as ufunc
    Parameters
    ----------
    x :

    Returns
    -------

    """
    # save the sign of x
    sign = x.copy()
    sign[x >= 0] = 1
    sign[x < 0] = - 1
    x = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(- x * x)
    return sign * y  # erf(-x) = -erf(x)


def rank_features_cross_sectionally(raw_signals: np.ndarray,
                                    date_ids: np.ndarray,
                                    ranking: str,
                                    axis_: int = 0,
                                    use_pandas: bool = True,
                                    print_time: bool = False):
    """
    Ranking features on each date. This is done in numpy for speed, but the drawback is that
    we assume dates are pre-ordered
    Parameters
    ----------
    raw_signals
    date_ids
    ranking
    axis_
    print_time : If true we print the run tiem
    use_pands :  IF true we run the pandas version.

    Returns
    -------

    """

    date_ids = np.array(date_ids)

    if date_ids is not None:
        """
        Important: the code assumes that date_ids are already sorted! 
        Pre-process the data so that date_ids are increasing !! 
        """
        # here it is important that dates are sorted because when we concatenate,
        # we do so in the order of dates
        if use_pandas:
            start = time.time()
            ranked_signals = pd.DataFrame(raw_signals)
            ranked_signals['date'] = date_ids
            ranked_signals = ranked_signals.groupby('date').rank(pct=True, axis=0) - 0.5
            ranked_signals = ranked_signals.values
            if print_time:
                print('Time pandas', time.time() - start)
        else:
            start = time.time()
            ranked_signals = np.concatenate([smart_ranking(raw_signals[date_ids.flatten() == date, :],
                                                           axis=axis_, ranking=ranking) for date in
                                             np.unique(date_ids.flatten())], axis=axis_)
            ranked_signals[np.isnan(raw_signals)] = np.nan
            if print_time:
                print('Time numpy', time.time() - start)
            # tmp = ranked_signals1[:, 0]
            # tmp1 = ranked_signals[:, 0]
            # joined = pd.DataFrame(np.concatenate([tmp.reshape(-1, 1), tmp1.reshape(-1, 1)], axis=1))
    else:
        ranked_signals = smart_ranking(raw_signals,
                                       axis=0,
                                       ranking=ranking)
    ranked_signals[np.isnan(raw_signals)] = np.nan

    return ranked_signals


def diagonal_shrinkage(sigma, lam):
    return (1 - lam) * sigma + np.diag(np.diag(sigma)) * lam


def markowitz(slice, expected_returns=None, lam=0.1):
    if expected_returns is None:
        expected_returns = slice.mean(0)
    if (slice ** 2).sum(0).sum() < 10 ** (-10):
        return (expected_returns * 0).reshape(-1, 1)
    indicator = (slice ** 2).sum(0) > 10 ** (-10)
    slice_ = slice[:, indicator]
    covar = np.linalg.inv(diagonal_shrinkage(np.matmul(slice_.T, slice_), lam))

    expected_returns = expected_returns.reshape(-1, 1)
    portfolio = np.matmul(covar, expected_returns[indicator.flatten(), :])
    full_portfolio = expected_returns * 0
    full_portfolio[indicator.flatten(), :] = portfolio
    return full_portfolio.reshape(1, -1)


def rolling_markowitz(returns,
                      expected_returns,
                      rolling_window=50,
                      lag=2):
    expected_returns = expected_returns.loc[expected_returns.isna().sum(1) < expected_returns.shape[1] - 3].fillna(0)
    returns = returns.reindex(expected_returns.index).fillna(0)
    rets = returns.values
    exps = expected_returns.values
    return_slices = [rets[i:min((i + rolling_window), returns.shape[0] - lag), :]
                     for i in range(returns.shape[0] - rolling_window - lag + 1)]
    portfolios = np.concatenate([markowitz(return_slices[i], exps[min((i + rolling_window),
                                                                      returns.shape[0] - lag), :])
                                 for i in range(returns.shape[0] - rolling_window - lag + 1)], axis=0)
    markowitz_portfolios = 0 * returns
    markowitz_portfolios.iloc[(rolling_window + lag - 1):returns.shape[0]] = portfolios
    returns_markowitz = (markowitz_portfolios * returns).sum(1)
    return returns_markowitz


def simple_markowitz(matrix: np.ndarray,
                     returns: np.ndarray,
                     shrinkage_list: list,
                     ready_means: np.ndarray=None):

    if ready_means is None:
        ready_means = returns.mean(0)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    multiplied = eigenvectors.T @ ready_means.reshape(-1, 1)
    normalizer = 1 / (eigenvalues.reshape(-1, 1) + np.array(shrinkage_list).reshape(1, -1))
    tmp = eigenvectors @ (multiplied.reshape(-1, 1) * normalizer)  # this is now P \times len(Z)
    return tmp


def long_only_markowitz(returns1):
    returns = returns1.dropna()
    returns = returns.loc[:, returns.std() > 0]
    if returns.shape[1] == 0:
        return 0, 0, 0
    mu = returns.mean(0).values.reshape(-1, 1)
    sigma = np.matmul(returns.values.T, returns.values)
    sigma_inv = np.linalg.inv(sigma)
    pi = np.matmul(sigma_inv, mu)
    if np.min(pi) < 0:
        return 0, 0, 0
    else:
        eff = 0.5 * np.matmul(mu.T, pi)
        rets = np.matmul(returns, pi.reshape(-1, 1)).sum(1)
        return eff, rets, pi


def best_long_only_efficient_portfolio(returns):
    num = len(returns.columns) + 1
    subsets = list()
    best_portfolio = pd.DataFrame(np.zeros([1, num - 1]), columns=returns.columns)
    if returns.std().sum() == 0:
        return best_portfolio, returns.mean(1)
    for ll in range(1, num):
        subsets = subsets + list(itertools.combinations(list(returns.columns), ll))
    efficiency = 0
    best_pi = None
    for set in subsets:
        eff, rets, pi = long_only_markowitz(returns[list(set)])
        if eff > efficiency:
            efficiency = eff
            best_set = set
            best_ret = rets
            best_pi = pi
    best_portfolio[list(best_set)] = best_pi.flatten()
    # best_portfolio1, best_ret1 = long_only_markowitz_with_cvx(returns)
    # print('the two sharpes are:-----\n ',
    #      sharpe_ratio(best_ret, horizon=256/12), sharpe_ratio(best_ret1, horizon=256/12), '\n-----------')
    return best_portfolio, best_ret


def add_data_frame_to_data_frame(df, df_to_be_added):
    if df.shape[0] == 0:
        df = df_to_be_added.fillna(0)
    else:
        tmp = pd.concat([df, df_to_be_added], axis=1)
        df = df_to_be_added.reindex(tmp.index).fillna(0) + df.reindex(tmp.index).fillna(0)

    return df.fillna(0)


def regression_with_tstats(predicted_variable, explanatory_variables):
    '''
    This function gets t-stats from regression
    :param predicted_variable:
    :param explanatory_variables:
    :return:
    '''
    x_ = explanatory_variables
    x_ = sm.add_constant(x_)
    y_ = predicted_variable
    # Newey-West standard errors with maxlags
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    try:
        tstat = np.round(result.summary2().tables[1]['z'], 1)  # alpha t-stat (because for 'const')
        tstat.index = list(z_.columns)
    except:
        print(f'something is wrong for t-stats')
    return tstat


def reduce_dimension_of_signals_and_returns(returns, signal_list, reduced_dimension, rolling_window=120):
    """
    pick top PCs of returns and then reduce dimensions of signals and returns
    :param returns:
    :param signal_list:
    :param reduced_dimension:
    :param rolling_window:
    :return:
    """

    transformed_r = returns.copy().fillna(0).iloc[:, -reduced_dimension:] * 0
    # tmp = transformed_r.iloc[:, :2].copy() * 0
    # populate zeros
    transformed_signals = list()
    for jj in range(len(signal_list)):
        signal_list[jj] = signal_list[jj].fillna(0)
        transformed_signals += [transformed_r.copy() * 0]

    previous_vectors = np.array([0])

    for ii in range(rolling_window + 1, returns.shape[0]):
        r_slice = returns.iloc[(ii - rolling_window):(ii - 1 + 1), :].fillna(0).values
        sigma = np.matmul(r_slice.T, r_slice)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        if np.sum(eigenvalues > 0.0001) < reduced_dimension:
            continue
        if previous_vectors.shape[0] == 1 and np.sum(eigenvalues) > 0:
            previous_vectors = eigenvectors
        else:
            sign_flips = np.sign(np.diag(np.matmul(previous_vectors.T, eigenvectors)))
            # print('flips', ii, np.sum(sign_flips < 0))
            eigenvectors = np.matmul(eigenvectors, np.diag(sign_flips))
            # print(pd.DataFrame(np.append(eigenvectors[:, sign_flips < 0][:, -2:], previous_vectors[:, sign_flips < 0][:, -2:], axis=1)))
            previous_vectors = eigenvectors
        v_vec = eigenvectors[:, -reduced_dimension:]
        r = returns.iloc[ii, :].values.reshape(-1, 1)
        transformed_r.iloc[ii, :] = np.matmul(v_vec.T, r).flatten()
        r1 = transformed_r.iloc[ii, :]

        for jj in range(len(signal_list)):
            s = signal_list[jj].iloc[ii, :].values.reshape(-1, 1)
            transformed_signals[jj].iloc[ii, :] = np.matmul(v_vec.T, s).flatten()
            s1 = transformed_signals[jj].iloc[ii, :]
        # tmp.iloc[ii, 0] = (r * s).sum()
        # tmp.iloc[ii, 1] = (r1 * s1).sum()

    for jj in range(len(signal_list)):
        transformed_signals[jj] = transformed_signals[jj].iloc[rolling_window:, :]
    return transformed_r.iloc[rolling_window:, :], transformed_signals  # , tmp


def sharpe_ratio(returns, horizon):
    """
    :param returns: returns: pandas DataFrame
    :return: annualized sharpe Ratio
    """
    if not isinstance(returns, pd.DataFrame) and not isinstance(returns, pd.Series):
        raise Exception('Returns must be a pandas dataframe')
    mult = 252 / horizon  # say, for monthly, horizon = 25, and we multiply by sqrt(10)
    sh = np.sqrt(mult) * returns.mean(0) / returns.std(0)
    # return np.round(sh, 2)
    return sh


def sharpe_ratio_by_freq(returns, freq):
    '''
    This function adjusts SR by frequency
    :param returns:
    :param freq: freq == 'daily', 'monthly', or 'annually'
    :return:
    '''
    if not isinstance(returns, pd.DataFrame) and not isinstance(returns, pd.Series):
        raise Exception('Returns must be a pandas dataframe')
    if freq == 'monthly':
        sh = np.sqrt(12) * returns.mean(0) / returns.std(0)
    elif freq == 'daily':
        sh = np.sqrt(252) * returns.mean(0) / returns.std(0)
    elif freq == 'annually':
        sh = returns.mean(0) / returns.std(0)
    elif freq == 'weekly':
        sh = np.sqrt(252 / 5) * returns.mean(0) / returns.std(0)
    elif freq == 'quarterly':
        sh = 2 * returns.mean(0) / returns.std(0)
    return sh  # np.round(sh, 2)


def demean_cross_section(data_frame: pd.DataFrame, standardize=False):
    # use numpy broadcasting for a truly Pythonic demeaning
    if not standardize:
        if len(data_frame.shape) > 1:
            return data_frame - data_frame.mean(axis=1).values.reshape(-1, 1)
        else:
            return data_frame - data_frame.mean()
    else:
        if len(data_frame.shape) > 1:
            return (data_frame - data_frame.mean(axis=1).values.reshape(-1, 1)) / data_frame.std(axis=1).values.reshape(
                -1, 1)
        else:
            return (data_frame - data_frame.mean()) / data_frame.std()


def vol_adjust_data(rets, periods=24, monthly=True) -> (np.ndarray, np.ndarray):
    """
    simple vol adjusted returns
    :param rets:
    :param periods:
    :return: tuple(vol_adjusted_returns, 1/vol)
    """
    if monthly:
        ss = 1
    else:
        ss = 2
    vol = rets.rolling(periods).std(min_periods=int(np.round(0.8 * periods))).shift(ss)
    vol = vol.clip(lower=vol.rolling(periods * 5,
                                     min_periods=int(0.6 * periods * 5)).quantile(0.1,
                                                                                  interpolation='higher'))

    vol[vol == 0] = np.nan
    adjustment = 1 / vol
    tmp1 = rets * adjustment
    return tmp1, adjustment


class TicToc:  # simple class to measure the time of various part of the code
    def __init__(self):
        self.start_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, message=''):
        t = time.time() - self.start_time
        print(f'{message}: in {np.round((t) / 60, 2)}m', flush=True)
        self.tic()
        return t


def get_all_random_matrix_quantities_for_the_panel(c_: float,
                                                   sigma_hat_eig: np.ndarray,
                                                   psi_hat_eig: np.ndarray,
                                                   z_grid: list,
                                                   port_ret_ins: np.ndarray = None,
                                                   save_sigma_and_psi_eig_in_rmt_stuff: bool = False):
    """
    This function computes all random matrix quantities from the paper
    Parameters
    ----------
    save_sigma_and_psi_eig_in_rmt_stuff : Boolean for saving
    z_grid :
    port_ret_ins : in sample portfolio returns; to be used later for bstar recovery
    c_ :
    sigma_hat_eig :
    psi_hat_eig :

    Returns
    -------

    """
    m_ = stieltjes_transform(z=-np.array(z_grid), psi_hat_eig=psi_hat_eig)

    xi = solve_for_xi(eigenvalues_of_sigma=sigma_hat_eig,
                      m_=m_,
                      z_=z_grid,
                      c_=c_,
                      )  # the equality portion of formula 34
    # maxit=10000,
    # tolerance=10 ** (-20)
    # now we compute xi_prime
    # numerator is P^{-1} trace( Psi / (zI+Psi)^2)
    numerator = - (((psi_hat_eig.reshape(1, -1) + np.array(z_grid).reshape(-1, 1)) ** (-2)) *
                   psi_hat_eig.reshape(1, -1)).mean(1)

    denominator = (((1 + xi.reshape(-1, 1) * sigma_hat_eig.reshape(1, -1)) ** (-2))
                   * sigma_hat_eig.reshape(1, -1)).mean(1) / c_

    xi_prime = numerator / denominator
    test = np.array(z_grid) * xi_prime + xi
    print(f'{test} must be positive ')

    nu = 1 - (1 / c_) * np.array(z_grid) * xi  # before I had psi_hat_eig.mean() instead of 1 here.

    # But this was wrong, as is explained in the paper, \xi already has \psi_{*,1} in it (divided by).
    nu = np.clip(nu, 0, 1)
    print(f'nu must be strictly positive and strictly smaller than 1: {nu}')

    nu_prime = - (1 / c_) * (xi + z_grid * xi_prime)
    nu_prime = np.clip(nu_prime, - 10 ** 10, 0)  # nu_prime is negative

    nu_hat = nu + z_grid * nu_prime
    nu_hat = np.clip(nu_hat, 0, 1)
    if port_ret_ins is None:
        return xi, xi_prime, nu, nu_prime, nu_hat

    trace_1 = np.sum(sigma_hat_eig.reshape(-1, 1) / (1 + xi.reshape(1, -1) * sigma_hat_eig.reshape(-1, 1)), 0)
    trace_2 = np.sum((sigma_hat_eig ** 2).reshape(-1, 1) / (1 + xi.reshape(1, -1) * sigma_hat_eig.reshape(-1, 1)), 0)

    b_star_hat = (port_ret_ins - xi * trace_1) / (xi * trace_2 + sum(sigma_hat_eig) * (1 - np.array(z_grid) * xi / c_))
    b_star_hat = np.maximum(b_star_hat, 0)

    if save_sigma_and_psi_eig_in_rmt_stuff:
        return b_star_hat, xi, nu, nu_prime, nu_hat, sigma_hat_eig, psi_hat_eig
    else:
        return b_star_hat, xi, nu, nu_prime, nu_hat


def solve_for_xi(eigenvalues_of_sigma: np.ndarray,
                 m_: np.ndarray,
                 z_: np.ndarray,
                 c_: float,
                 tolerance: float = 1e-10,
                 maxit: int = 100) -> np.ndarray:
    """
    According to our paper, xi(z;c) is the unique solution to the equation
    c_ * (1 - z_ * m(-z_;c_)) - 1 = - N^{-1} * tr((I+xi(z)\Sigma)^{-1})
    Note that
    f(xi) = tr((I+xi\Sigma)^{-1}) = \sum_i (1+xi * lambda(i))^{-1}
    and hence
    f'(xi) = - \sum_i (1+xi * lambda(i))^{-2}
    and we will use these explicit expressions in Newton's method.

    In most calculations, we will have c_ quite small because c_ = P / (T * N).
    In this case, xi(z) will also be small, and we can use the approximation
    (I+xi(z)\Sigma)^{-1} \approx\ I - xi \Sigma to get
    c_ * (1 - z_ * m(-z_;c_)) - 1 ~ - N^{-1}tr  (I - xi \Sigma) = -1 + xi * N^{-1} tr(Sigma)
    which means
    xi ~ c_ * (1 - z_ * m(-z_;c_)) / (N^{-1} tr(Sigma))
    We will use this as the starting point

    Parameters
    ----------
    tolerance :
    c_ :
    z_ :
    eigenvalues_of_sigma :
    m_ :
    maxit:
    Returns
    -------
    """
    eigenvalues_of_sigma = eigenvalues_of_sigma.flatten()

    # use the theory above to guess the starting point
    starting_point = c_ * (1 - z_ * m_) / np.mean(eigenvalues_of_sigma)

    solution = starting_point

    # Newton method: The beauty is that we are solving it in one go for all values of z_ !
    # so we do not need to loop over different values of z_
    iter = 0
    error = np.array(10 ** 10)
    while iter < maxit and np.max(np.abs(error)) > tolerance:
        # I am normalizing everything by c_ which is small; otherwise it will converge too fast
        # matrix multiplication to compute mean(z*eig_sigma) for different z
        error = (c_ * (1 - z_ * m_) - 1 + np.mean(1 /
                                                  (1 + solution.reshape(1, -1).T @ eigenvalues_of_sigma.reshape(1, -1)),
                                                  axis=1)) / c_
        error_slope = - np.mean(eigenvalues_of_sigma /
                                ((1 + solution.reshape(1, -1).T @ eigenvalues_of_sigma.reshape(1, -1)) ** 2),
                                axis=1) / c_
        solution -= error / error_slope
        # if iter % 100 == 0:
        # print(f'iteration {iter}. solution = {solution}. error = {error}')
        iter += 1
    return solution


def stieltjes_transform(z: float, psi_hat_eig: np.ndarray):
    """
    Function m(-z;c) = P^{-1} tr((z I + \hat{\Psi})^{-1}). Using eigenvalues of psi_hat this is
    m(-z; c) = P^{-1} sum_i 1 / (z + psi_hat_eig(i))

    :param z:
    :param psi_hat_eig:
    :return:
    """
    p_ = psi_hat_eig.shape[0]
    return np.sum(1 / (psi_hat_eig.reshape(-1, 1) - z.reshape(1, -1)), axis=0) / p_


def build_managed_returns_for_msrr(random_features,
                                   y_train,
                                   factor_msrr,
                                   date_indices_for_panel):
    random_features = random_features * y_train.reshape(-1, 1)  # these are actually managed returns, R * F(S)
    if factor_msrr:
        # it is here where factors are constructed, summing across tickers
        random_features = pd.DataFrame(random_features)
        random_features['dates'] = date_indices_for_panel
        random_features = random_features.groupby('dates').sum().values
    return random_features
