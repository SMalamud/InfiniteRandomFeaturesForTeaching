import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import basic_parameters.basic_random_features_settings_for_futures
from helpers.auxilliary_functions \
    import vol_adjust_data, demean_cross_section, sharpe_ratio, add_data_frame_to_data_frame


class Futures:
    """
    

    """

    folder = 'FUTURES'
    if not os.path.exists(folder):
        os.mkdir(folder)
    # build path for plots
    plots_folder = os.path.join(folder, 'plots')
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)

    def __init__(self,
                 horizon,
                 clip_extreme_returns=False,
                 get_summary_statistics=False,
                 window_for_signal_volatility=60,
                 vol_standardize_all_signals=True,
                 gamma_list_str=None,
                 momentum_windows=[21, 63, 126, 252],
                 use_binning=False,
                 add_constant_factor=None,
                 merge_CME_SP_and_CME_ES_by_dollar_volume=False):
        """
        Initialization function for class Futures
        :param horizon:
        :param clip_extreme_returns:
        :param get_summary_statistics:
        :param window_for_signal_volatility:
        :param vol_standardize_all_signals:
        :param gamma_list_str: gamma_list_str is used in the horizon results folder
        """

        self.simple_mom_and_carry_returns = None
        settings = \
            basic_parameters.basic_random_features_settings_for_futures.BasicRandomFeatureSettingForFutures(
                horizon=horizon,
                window_for_signals_volatility=window_for_signal_volatility,
                gamma_list_str=gamma_list_str,
                use_binning=use_binning,
                add_constant_factor=add_constant_factor)

        self.settings = settings
        # create an instance object
        file = os.path.join(self.folder, 'continuous_futures.npy')
        if not os.path.exists(file):
            self.create_futures()
        dollar_volume_file = os.path.join(self.folder, 'continuous_futures_dollar_volume.npy')
        if not os.path.exists(dollar_volume_file):
            self.create_futures_dollar_volume()
        self.vol_standardize_all_signals = vol_standardize_all_signals
        self.horizon = horizon  # this is the horizon (frequency) of returns
        # data is then also assumed to be observed only at this frequency
        self.metadata = pd.read_csv(os.path.join(self.folder, 'futures_metadata.csv'))

        self.universes = None  # universes are ['bond', 'commodity', 'currency', 'rate', 'stock', 'full_futures']
        self.universe_lists = None  # true symbols in each universe
        self.basis = None  # basis = futures_price / spot_price, dictionary
        self.returns = None  # returns, dim is dates times ticker (15990 \times 79)
        self.all_signals = None  # signals, dictionary
        # dict_keys([('mom', 32), ('basis', 32), ('mom', 64), ('basis', 64),
        # ('mom', 128), ('basis', 128), ('mom', 256), ('basis', 256)])
        # each element has dimension dates times ticker (15990 \times 79)

        self.window_for_return_volatility = settings.window_for_return_volatility
        self.window_for_signals_volatility = settings.window_for_signals_volatility
        self.long_term_window_for_volatility = settings.long_term_window_for_volatility
        self.return_signal_lag = settings.return_signal_lag
        self.add_constant_factor = settings.add_constant_factor

        data = np.load(file, allow_pickle=True)[()]
        dollar_volume_data = np.load(dollar_volume_file, allow_pickle=True)[()]
        # [()] change the data structure from 'numpy.ndarray' object to dictionary
        reindexed_data = dict()
        reindexed_dollar_volume_data = dict()
        # Change the index of prices dataframe
        for key in data:
            prices = data[key].copy()  # dataframe of prices
            if prices.shape[0] == 0:
                continue
            prices.index = pd.to_datetime(prices.index)  # set the index as date
            reindexed_data[key] = prices.fillna(method='ffill')

            dollar_volumes = dollar_volume_data[key].copy()  # dataframe of prices
            if dollar_volumes.shape[0] == 0:
                continue
            dollar_volumes.index = pd.to_datetime(dollar_volumes.index)  # set the index as date

            # check dimension of dollar_volumes.index with prices.index
            if not len(prices.index) == len(dollar_volumes.index):
                print('Dollar volume index does not match the price index!!!')
                breakpoint()
            reindexed_dollar_volume_data[key] = dollar_volumes.fillna(method='ffill')

        for key in data:
            if not key in reindexed_data.keys():
                continue
            if key[0] > 1:
                prices = reindexed_data[key]
                prices = prices.reindex(reindexed_data[(1, key[1])].index)
                # to make sure longer-term guys live on the same dates as front month
                reindexed_data[key] = prices.fillna(method='ffill')

                dollar_volumes = reindexed_dollar_volume_data[key]
                dollar_volumes = dollar_volumes.reindex(reindexed_dollar_volume_data[(1, key[1])].index)
                # to make sure longer-term guys live on the same dates as front month
                reindexed_dollar_volume_data[key] = dollar_volumes.fillna(method='ffill')

        self.raw_prices = reindexed_data  # this is data at daily frequency
        self.raw_dollar_volume = reindexed_dollar_volume_data
        ##############################################################################
        #########################################################################################
        # !!!!!!!!!!!!!!!!!!!!!!!!! BUILDING DAILY STUFF  !!!!!!!!!!!!!!!!!!!!!!
        # basis is daily, lagged by 1 day
        vol_normalizer = self.build_tradable_back_adjusted_returns(
            ready_vol_normalizer=None,
            clip_extremes=clip_extreme_returns,
            get_summary_statistics=get_summary_statistics,
            merge_CME_SP_and_CME_ES_by_dollar_volume=merge_CME_SP_and_CME_ES_by_dollar_volume)  # self.returns
        self.build_basis(
            merge_CME_SP_and_CME_ES_by_dollar_volume=merge_CME_SP_and_CME_ES_by_dollar_volume)  # build self.basis at daily frequency, ALREADY LAGGED BY 1 DAY !!!
        # now we have produced daily 1/vol. We will also use it to normalize horizon returns below
        #######################################################################################
        # DAILY RETURNS LAGGED BY 1 DAY
        self.lagged_returns = self.returns.shift(1).copy()  # so these are daily returns

        # we are still at daily frequency. Hence, everything is in days!
        # self.momentum_windows = np.array([5, 21, 63, 126, 252]).astype(int)  # 1 month, 6 months, 12 months
        # self.momentum_windows = np.array([5, 21, 42]).astype(int)  # CBOE_VX
        self.momentum_windows = np.array(momentum_windows).astype(int)  # 1 month, 6 months, 12 months
        # NOW WE BUILD ALL SIGNALS AT DAILY FREQUENCY, AND THEY ARE LAGGED BY 1 DAY
        self.build_futures_signals(self.momentum_windows)  # these are daily signals

        ###########################################################################################################
        # now we move the horizon frequency
        # REINDEX !!!
        #######################################################################################
        # reindexing block !!!!!
        self.raw_prices = {key: self.raw_prices[key].iloc[0::horizon] for key in self.raw_prices}
        self.raw_dollar_volume = {key: self.raw_dollar_volume[key].iloc[0::horizon] for key in self.raw_prices}
        self.basis = {key: self.basis[key].reindex(self.raw_prices[(1, key[1])].index) for key in self.basis}
        self.all_signals = {key: self.all_signals[key].reindex(self.raw_prices[(1, 'back_adjusted')].index)
                            for key in self.all_signals}

        vol_normalizer = vol_normalizer.iloc[0::horizon] / np.sqrt(horizon)
        # NOW RETURNS ARE COMPUTED AS QUOTIENTS OF PRICES AT HORIZON FREQUENCY

        # we normalize horizon returns using vol_normalizer built from daily returns
        self.build_tradable_back_adjusted_returns(ready_vol_normalizer=vol_normalizer,
                                                  clip_extremes=clip_extreme_returns,
                                                  merge_CME_SP_and_CME_ES_by_dollar_volume=merge_CME_SP_and_CME_ES_by_dollar_volume)
        #######################################################################################
        self.build_universes(
            merge_CME_SP_and_CME_ES_by_dollar_volume=merge_CME_SP_and_CME_ES_by_dollar_volume)  # build self.universes

    def build_universes(self,
                        merge_CME_SP_and_CME_ES_by_dollar_volume=False):
        '''
        This function gets all true symbol for universes ['bond', 'commodity', 'currency', 'rate', 'stock', 'full_futures']
        :return:
        '''
        metadata = self.metadata
        metadata['true_symbol'] = metadata['Exchange'] + '_' + metadata['Symbol']

        if merge_CME_SP_and_CME_ES_by_dollar_volume:
            index_of_CME_ES = metadata.loc[metadata['true_symbol'] == 'CME_ES'].index
            metadata.loc[index_of_CME_ES, 'true_symbol'] = 'CME_SP_ES'

        futures = self.raw_prices
        symbols = futures[(1, 'raw')].columns.tolist()
        metadata.set_index(metadata.true_symbol, inplace=True)
        metadata.reindex(symbols)

        if merge_CME_SP_and_CME_ES_by_dollar_volume:
            metadata = metadata.drop('CME_SP')
            self.metadata = metadata

        self.universes = np.unique(metadata['universe']).tolist() + ['full_futures']
        # ['bond', 'commodity', 'currency', 'rate', 'stock', 'full_futures']

        # get true_symbols for each universe
        universe_lists = {universe: metadata.loc[metadata.universe == universe, 'true_symbol'].tolist() for
                          universe in self.universes}
        symbols = metadata['true_symbol'].tolist()

        # assign full_futures into universe_lists
        universe_lists.update({'full_futures': list(set(symbols) - set(universe_lists['rate']))})
        self.universe_lists = universe_lists

    def build_basis(self, depth1=2, depth2=1, merge_CME_SP_and_CME_ES_by_dollar_volume=False):
        '''
        This function builds basis = futures_price / spot_price according to
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2298565
        Since we do not observe spot, we proxy with nearest maturity futures contract
        :param depth1: maturity of the futures
        :param depth2: maturity of the spot
        :return:
        '''
        futures = self.raw_prices
        basis = dict()
        for type in ['raw', 'back_adjusted']:
            # signal = futures[(depth1, type)] / futures[(depth2, type)]
            # # we clip basis to get rid of outliers
            # signal = - (signal.copy() - 1).clip(-1, 1).fillna(method='ffill')
            signal = futures[(depth2, type)] / futures[(depth1, type)]
            # we clip basis to get rid of outliers
            signal = (signal.copy() - 1).clip(-1, 1).fillna(method='ffill')
            if merge_CME_SP_and_CME_ES_by_dollar_volume:
                dates_using_CME_ES = self.dates_using_CME_ES
                CME_SP_signal = signal['CME_SP']
                CME_ES_signal = signal['CME_ES']
                CME_SP_signal[dates_using_CME_ES] = CME_ES_signal[dates_using_CME_ES]

                signal['CME_SP'] = CME_SP_signal
                signal.rename(columns={'CME_SP': 'CME_SP_ES'}, inplace=True)
                signal = signal.drop(['CME_ES'], axis=1)
            basis[depth1, type] = signal.shift(1)

        self.basis = basis

    def build_tradable_back_adjusted_returns(self,
                                             get_summary_statistics=False,
                                             ready_vol_normalizer=None,
                                             clip_extremes=True,
                                             merge_CME_SP_and_CME_ES_by_dollar_volume=False):
        '''
        This function builds tradable back adjusted returns
        :return:
        '''
        futures = self.raw_prices
        futures_dollar_volume = self.raw_dollar_volume

        prices = futures[(1, 'back_adjusted')]  # Use back-adjusted price
        dollar_volumes = futures_dollar_volume[(1, 'back_adjusted')]

        if get_summary_statistics:
            produce_table_of_summary_statistics(prices, self.folder)

        returns = (prices / prices.shift(1) - 1)
        # breakpoint()
        # returns[['CME_SP', 'CME_ES']].corr()
        if merge_CME_SP_and_CME_ES_by_dollar_volume:
            print('Note that CME_ES is spliced with CME_SP by the following two steps: ')
            print('Step 1: Set the splice date to the date when CME_ES volume overtook CME_SP volume.')
            print('Step 2: The notional of e-mini is $50, and notional of $250. '
                  'So we compare ES dollar volume against 1/5 SP dollar volume')

            # fill-in nan value, but returns['CME_SP'] doesn't change in the following three lines
            # because CME_ES has more NaN than CME_SP
            CME_SP_returns = returns['CME_SP']
            CME_ES_returns = returns['CME_ES']
            dates_CME_SP_returns_nan_index = CME_SP_returns[CME_SP_returns.isna()].index

            # Set the splice date to the date when CME_ES dollar volume overtook 1/5 SP CME_SP dollar volume
            dates_using_CME_ES = dollar_volumes['CME_SP'][dollar_volumes['CME_SP'] < 5 * dollar_volumes['CME_ES']].index
            dates_using_CME_ES = dates_using_CME_ES.union(dates_CME_SP_returns_nan_index)
            CME_SP_returns[dates_using_CME_ES] = CME_ES_returns[dates_using_CME_ES]
            self.dates_using_CME_ES = dates_using_CME_ES

            # merge into returns
            returns['CME_SP'] = CME_SP_returns
            returns.rename(columns={'CME_SP': 'CME_SP_ES'}, inplace=True)
            returns = returns.drop(['CME_ES'], axis=1)

        returns = returns.fillna(0)

        # save the raw returns
        save_returns = False
        if save_returns:
            returns.to_csv(os.path.join(self.folder, 'futures_back_adjusted_returns.csv'), index=True)

        returns[returns.abs() > 1] = 0  # returns of more than 100% are most probably data issues

        # vol is computed by wind
        if ready_vol_normalizer is None:
            wind = self.window_for_return_volatility
            vol = np.sqrt((returns ** 2).rolling(wind, min_periods=int(0.6 * wind)).mean())
            lower_clip = vol.rolling(self.long_term_window_for_volatility,
                                     min_periods=int(0.6 * self.long_term_window_for_volatility)) \
                .quantile(0.1, interpolation='higher')
            # drop the vol lower than quantile 0.1 in the past self.long_term_window_for_volatility days
            vol = vol.clip(lower=lower_clip)
            vol_normalizer = 1 / vol.shift(self.return_signal_lag + 1)
        else:
            vol_normalizer = ready_vol_normalizer

        # Normalize returns by lag-2 vol
        # With daily data, shifting by 1 day is not tradable
        # Say you observe a signal (even a volatility ) at time t. But we only have closing prices.
        # By the time you observed it, the market is closed, so you can only trade on this information next day
        # Then the return you realize is r_{t+1} * signal(t-1), and r_{t+1} = (price(t+1)/price(t) - 1)
        full_returns = returns * vol_normalizer
        full_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

        # winsorize at 99.5 and 0.5 quantile by tickers
        if clip_extremes:
            quantile_995 = full_returns.quantile(99.5 * 0.01, axis=0)
            quantile_005 = full_returns.quantile(0.5 * 0.01, axis=0)
            full_returns = full_returns.clip(lower=quantile_005, upper=quantile_995)
        self.returns = full_returns

        # remove abnormal returns over 100
        full_returns[full_returns.abs() > 100] = 0

        return vol_normalizer

    def build_futures_signals(self, mom_windows, build_signals_on_mom_windows=True):
        '''
        This function build the future signals by momentum rolling windows
        :param mom_windows: momentum rolling windows
        :return:
        '''
        all_signals = dict()
        # WE ARE USING LAGGED RETURNS, SO THAT THE DAILY SIGNALS ARE ALWAYS LAGGED BY 1 DAY
        lagged_returns = self.lagged_returns
        basis = self.basis[(2, 'raw')]  # raw basis for the front and second month, already LAGGED BY 1 DAY in
        # function build_basis()

        signals_vol_wind = self.window_for_signals_volatility
        long_term_window_for_volatility = self.long_term_window_for_volatility

        if build_signals_on_mom_windows:
            for mom_window in mom_windows:
                mom_signals = lagged_returns.rolling(mom_window, min_periods=int(0.8 * mom_window)).mean()
                basis_signals = basis.rolling(mom_window, min_periods=int(0.8 * mom_window)).mean()

                if self.vol_standardize_all_signals:
                    all_signals['mom', mom_window] = signals_vol_standardization(mom_signals,
                                                                                 signals_vol_wind,
                                                                                 long_term_window_for_volatility)
                    all_signals['basis', mom_window] = signals_vol_standardization(basis_signals,
                                                                                   signals_vol_wind,
                                                                                   long_term_window_for_volatility)
                else:
                    all_signals['mom', mom_window] = mom_signals
                    all_signals['basis', mom_window] = basis_signals
        else:
            # build signals on monthly lags
            max_mom_windows = np.max(mom_windows)
            window_for_each_lag = 21
            number_on_monthly_lags = int(max_mom_windows / window_for_each_lag)
            for monthly_lag in range(number_on_monthly_lags):
                # signals are built by lagged returns from monthly_lag to monthly_lag+1
                daily_lag = monthly_lag * window_for_each_lag
                returns_lag_temp = lagged_returns.shift(daily_lag)
                mom_signals = returns_lag_temp.rolling(window=number_on_monthly_lags,
                                                       min_periods=int(0.8 * number_on_monthly_lags)).mean()
                basis_signals = basis.shift(daily_lag).rolling(window=number_on_monthly_lags,
                                                               min_periods=int(0.8 * number_on_monthly_lags)).mean()

                if self.vol_standardize_all_signals:
                    all_signals['mom', daily_lag + window_for_each_lag] = signals_vol_standardization(mom_signals,
                                                                                                      signals_vol_wind,
                                                                                                      long_term_window_for_volatility)
                    all_signals['basis', daily_lag + window_for_each_lag] = signals_vol_standardization(basis_signals,
                                                                                                        signals_vol_wind,
                                                                                                        long_term_window_for_volatility)
                else:
                    all_signals['mom', daily_lag + window_for_each_lag] = mom_signals
                    all_signals['basis', daily_lag + window_for_each_lag] = basis_signals
        self.all_signals = all_signals  # in daily frequency

    def build_momentum_and_carry_returns(self, plot=False):
        '''
        This function construct momentum and carry returns for each universes
        :param plot: whether to plot or not
        :return:
        '''
        rank = True
        all_rets = {key: pd.DataFrame() for key in self.all_signals}
        all_signals = self.all_signals  # in the frequency of horizon

        for universe in self.universes:
            subset = self.universe_lists[universe]
            rets = self.returns[subset]

            for sig_name in all_signals:
                signal = all_signals[sig_name]  # gets data for sig_name

                # signal cross-sectional demeaning to make the strategy purely cross-sectional
                if rank:
                    signal = rank_futures(signal, fillnans=True)
                else:
                    signal = demean_cross_section(signal)

                # Signals has already been lagged by 1
                signal = signal.fillna(0).shift(self.return_signal_lag)  # signals at t or t-1 depending on whether
                # we have shifted signals by 1 day already
                retts = rets.fillna(0)  # returns at t+1

                trading_strategy_return = (retts * signal).sum(1)

                all_rets[sig_name][universe] = vol_adjust_data(trading_strategy_return,
                                                               self.window_for_return_volatility, monthly=False)[0]

        # tmp = 0 * all_rets[('mom', self.momentum_windows[0])]
        first_key = list(all_rets.keys())[0]
        tmp = 0 * all_rets[first_key]
        for sig_name in all_signals:
            tmp = add_data_frame_to_data_frame(tmp, all_rets[sig_name])
        all_rets['all_signals_together'] = tmp

        self.simple_mom_and_carry_returns = all_rets
        if plot:
            for sig_name in list(all_signals.keys()) + ['all_signals_together']:
                plt.figure()
                all_rets[sig_name].cumsum().plot(title=f'{sig_name}')
                plt.savefig(os.path.join(self.plots_folder, f'futures_{sig_name}_{self.horizon}.jpeg'))
                print(sig_name, sharpe_ratio(all_rets[sig_name], self.horizon))
                if sharpe_ratio(all_rets[sig_name], self.horizon).isna().sum() > 0:
                    breakpoint()
                plt.close('all')
        return all_rets

    def get_all_signals_for_a_ticker(self, ticker):
        '''
        This function gets all signals for a specific ticker
        IMPORTANT: IT IS HERE WHERE WE LAG THE SIGNALS
        :param ticker: Ticker to get all signals
        :return:
        '''
        all_signals = self.all_signals

        all_signals_for_ticker = pd.DataFrame()
        for key in all_signals:
            all_signals_for_ticker[key] = all_signals[key][ticker]

        # shift signals by lag with returns
        all_signals_for_ticker = all_signals_for_ticker.shift(self.return_signal_lag)

        if self.add_constant_factor is not None:
            all_signals_for_ticker['constant'] = self.add_constant_factor
        return all_signals_for_ticker


def rank_futures(data_frame, fillnans=True):
    """
    ranks signals
    :param data_frame: Data input in the structure of dataframe
    :param good_columns: columns to be ranked
    :param fillnans: whether we fill Na with zeros
    :return:
    """
    data = data_frame.copy()

    if fillnans:
        data = data.rank(pct=True, axis=1).fillna(0.5) - 0.5
    else:
        data = data.rank(pct=True, axis=1) - 0.5
    return data


def signals_vol_standardization(signals,
                                signals_vol_wind,
                                long_term_window_for_volatility):
    """
    This function standardize signals by it's time-rolling vols
    :param signals:
    :param signals_vol_wind: volatility window
    :param long_term_window_for_volatility: window to delete outliers
    :return:
    """
    if signals_vol_wind == 'expanding':
        signals_vol = np.sqrt((signals ** 2).expanding(min_periods=36 * 21).mean())
        # expanding vol starts from 36 months
    else:
        signals_vol = np.sqrt((signals ** 2).rolling(signals_vol_wind, min_periods=int(0.8 * signals_vol_wind)).mean())
        signals_vol = signals_vol.clip(lower=signals_vol.rolling(long_term_window_for_volatility)
                                       .quantile(0.1, interpolation='higher'))
    # clipping in the next line is a good idea after normalization 5 sigma realization is very unlikely
    signals = (signals / signals_vol.shift(2)).clip(-5, 5)
    return signals


def get_raw_signals(futures, signal_name, ticker):
    if signal_name == 'attention':
        tmp = futures.attention - futures.attention.rolling(24).mean()
    elif signal_name == 'mom_and_carry':
        """
        IMPORTANT: IT IS HERE WHERE WE LAG THE SIGNALS inside get_all_signals_for_a_ticker
        """
        tmp = Futures.get_all_signals_for_a_ticker(futures,
                                                   ticker)
    return tmp


def produce_table_of_summary_statistics(prices, folder):
    '''
    This function gets summary statistics of futures
    :param prices:
    :param folder:
    :return:
    '''
    # get summary statistics
    returns = (prices / prices.shift(1) - 1).fillna(np.nan)
    annualized_mean = returns.mean(0) * 252
    annualized_vol = returns.std(0) * np.sqrt(252)

    futures_info = pd.read_csv(os.path.join(folder, 'futures_metadata.csv'))
    futures_info['true_symbol'] = futures_info['Exchange'] + '_' + futures_info['Symbol']
    futures_info = futures_info.set_index('true_symbol')
    futures_info['annualized_mean'] = annualized_mean
    futures_info['annualized_vol'] = annualized_vol
    futures_info['annualized_SR'] = annualized_mean / annualized_vol
    futures_info['Start Date'] = pd.to_datetime(futures_info['Start Date'], format='%d.%m.%y').dt.strftime('%m/%d/%Y')

    # pd.to_datetime(futures_info['Start Date']).dt.strftime('%m/%d/%Y')
    pd.to_datetime(futures_info['Start Date']).dt.strftime('%b-%Y')
    futures_info = futures_info.drop(['Symbol', 'Globex Symbol', 'First Contract', 'Months', 'Number of Contracts',
                                      'Contract Size', 'Tick Size', 'Pricing Unit', 'Deliverable', 'Big Point Value'],
                                     axis=1)
    futures_info = futures_info.sort_values(by=['universe', 'Exchange'])

    futures_info = futures_info.loc[futures_info['Exchange'].isin(['CME', 'CBOE'])]
    futures_info.to_csv(os.path.join(folder, 'futures_metadata_summary_statistics.csv'), index=False)
