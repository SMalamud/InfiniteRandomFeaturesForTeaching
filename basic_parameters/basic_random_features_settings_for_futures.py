import numpy as np
import os


class BasicRandomFeatureSettingForFutures:
    def __init__(self,
                 horizon=1,
                 window_for_signals_volatility='expanding',
                 gamma_list_str=None,
                 use_binning=False,
                 add_constant_factor=None):
        """
        this function creates an instance of random features
        :param raw_signals:
        :param returns:
        :param balanced:
        """

        # create an instance object
        if horizon < 4:
            self.windows = [32, 64, 128, 256, 512, 1024]
            self.step_map = {12: 1, 16: 1, 32: 8, 64: 16, 128: 32, 256: 32, 512: 32, 1024: 64, 2048: 128}
            # self.window_for_vol_standardization = 60
            self.lag = 2  # this is the lag for computing regression coefficients
            self.returns_frequency = 'daily'
        elif (horizon > 4) and (horizon < 15):  # horizon = 5 is weekly frequency
            # self.windows = [16, 32, 64, 128, 256, 512]
            self.windows = [12, 60, 120] # [16, 64, 128]
            # self.step_map = {12: 1, 16: 1, 32: 4, 64: 8, 128: 16, 256: 32, 512: 32, 1024: 64, 2048: 128}
            self.step_map = {12: 1, 16: 1, 32: 1, 60: 1, 64: 1, 120: 1, 240: 1, 480: 1, 1024: 1, 2048: 1}
            # self.window_for_vol_standardization = 12
            self.lag = 2  # this is the lag for computing regression coefficients
            self.returns_frequency = 'weekly'
        elif horizon > 15:  # 21 is monthly
            self.windows = [12, 60, 120] #  [12, 16, 32, 64, 128, 256]  # up to 20 years
            self.step_map = {12: 1, 16: 1, 32: 1, 60: 1, 64: 1, 120: 1, 128: 1, 256: 1}
            # self.window_for_vol_standardization = 12
            self.lag = 1  # this is the lag for computing regression coefficients
            self.returns_frequency = 'monthly'

        self.window_for_return_volatility = 60  # days
        self.window_for_signals_volatility = window_for_signals_volatility
        self.long_term_window_for_volatility = 252  # 250 days
        # if self.window_for_signals_volatility = 'expanding', then we don't need self.long_term_window_for_volatility
        self.return_signal_lag = 1  # ALL SIGNALS ARE AUTOMATICALLY LAGGED BY 1 DAY + return_signal_lag
        # IMPORTANT: return_signal_lag IS IN THE UNITS OF horizon !!

        # self.numbers_of_signals = np.array([2, 4, 8, 12, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
        #                                     640, 768, 896, 1024, 2048, 4096])
        # self.numbers_of_signals = np.array([2, 4, 8, 12, 16, 24, 48, 60, 72, 84, 96, 108, 120, 160, 240, 360, 480, 720,
        #                                     960, 1200, 2400, 4800, 9600, 12000])
        self.numbers_of_signals = np.array([2, 4, 8, 12, 16, 24, 48, 60, 72, 84, 96, 108, 120, 160, 240, 360, 480, 720,
                                            960, 1200, 2400, 4800, 9600, 12000])
        # self.numbers_of_signals = np.array([5000])
        self.shrinkage_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
        self.use_msrr = False  # then we run simple ridge
        self.data_folder = '../STEVENS_FUTURES'
        self.gamma_list_str = gamma_list_str
        self.use_binning = use_binning
        self.add_constant_factor = add_constant_factor

        self.results_folder_string = 'WeeklyMom' # 'June10_returns_not_clipped'  # 'NoWeeklyMom'  # 'WeeklyMom'
        self.results_folder = os.path.join(self.data_folder, f'results_%s_gamma%s_binning%s_const%s'%(self.results_folder_string, gamma_list_str,use_binning, add_constant_factor))

        self.linear_plots_folder = os.path.join(self.data_folder, f'linear_managed_returns_%s'%(self.results_folder_string))
        if not os.path.exists(self.linear_plots_folder):
            os.mkdir(self.linear_plots_folder)

        self.verbose = True
        self.predict_one_period = False
        if 'semyon' in os.path.expanduser('~'):
            self.data_folder = '/hdd/semyon'
            self.results_folder = os.path.join(self.data_folder, f'results_random_features')
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)
        self.horizon = horizon



