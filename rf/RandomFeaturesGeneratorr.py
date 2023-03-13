import pandas as pd
import numpy as np
from typing import Optional, Tuple

from helpers.auxilliary_functions import rank_features_cross_sectionally, build_managed_returns_for_msrr


class RandomFeaturesGeneratorr:
    """
    This class manages creation of random features

    #########################################
    The list of possible distributions is:
    #########################################
    beta(a, b[, size])	Draw samples from a Beta distribution.
    binomial(n, p[, size])	Draw samples from a binomial distribution.
    chisquare(df[, size])	Draw samples from a chi-square distribution.
    dirichlet(alpha[, size])	Draw samples from the Dirichlet distribution.
    exponential([scale, size])	Draw samples from an exponential distribution.
    f(dfnum, dfden[, size])	Draw samples from an F distribution.
    gamma(shape[, scale, size])	Draw samples from a Gamma distribution.
    geometric(p[, size])	Draw samples from the geometric distribution.
    gumbel([loc, scale, size])	Draw samples from a Gumbel distribution.
    hypergeometric(ngood, nbad, nsample[, size])	Draw samples from a Hypergeometric distribution.
    laplace([loc, scale, size])	Draw samples from the Laplace or double
    exponential distribution with specified location (or mean) and scale (decay).
    logistic([loc, scale, size])	Draw samples from a logistic distribution.
    lognormal([mean, sigma, size])	Draw samples from a log-normal distribution.
    logseries(p[, size])	Draw samples from a logarithmic series distribution.
    multinomial(n, pvals[, size])	Draw samples from a multinomial distribution.
    multivariate_normal(mean, cov[, size, …)	Draw random samples from a multivariate normal distribution.
    negative_binomial(n, p[, size])	Draw samples from a negative binomial distribution.
    noncentral_chisquare(df, nonc[, size])	Draw samples from a noncentral chi-square distribution.
    noncentral_f(dfnum, dfden, nonc[, size])	Draw samples from the noncentral F distribution.
    normal([loc, scale, size])	Draw random samples from a normal (Gaussian) distribution.
    pareto(a[, size])	Draw samples from a Pareto II or Lomax distribution with specified shape.
    poisson([lam, size])	Draw samples from a Poisson distribution.
    power(a[, size])	Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
    rayleigh([scale, size])	Draw samples from a Rayleigh distribution.
    standard_cauchy([size])	Draw samples from a standard Cauchy distribution with mode = 0.
    standard_exponential([size])	Draw samples from the standard exponential distribution.
    standard_gamma(shape[, size])	Draw samples from a standard Gamma distribution.
    standard_normal([size])	Draw samples from a standard Normal distribution (mean=0, stdev=1).
    standard_t(df[, size])	Draw samples from a standard Student’s t distribution with df degrees of freedom.
    triangular(left, mode, right[, size])	Draw samples from the triangular distribution over the interval [left, right].
    uniform([low, high, size])	Draw samples from a uniform distribution.
    vonmises(mu, kappa[, size])	Draw samples from a von Mises distribution.
    wald(mean, scale[, size])	Draw samples from a Wald, or inverse Gaussian, distribution.
    weibull(a[, size])	Draw samples from a Weibull distribution.
    zipf(a[, size])	Draw samples from a Zipf distribution.

    ####################################################################################################################
    activation functions
    ####################################################################################################################
    cos, sin, exp, arctan (sigmoid style!), tanh,
    ReLu (careful, relu is implemented separately through the multiplication method (x * (x > 0)) which is the fastest
    Elu (x * (x > 0)) + alpha * (exp(x) - 1) * (x < 0)
    and SoftPlus = log(1+exp(x))

    """

    # the next shows the number of basic_parameters defining the distribution
    distribution_requirements = {
        "beta": 2,
        "binomial": 2,
        "chisquare": 0,
        "dirichlet": 1,
        "exponential": 0,
        "f": 2,
        "gamma": 1,
        "geometric": 1,
        "gumbel": 2,
        "hypergeometric": 2,
        "laplace": 2,
        "logistic": 2,
        "lognormal": 2,
        "logseries": 1,
        "multinomial": 2,
        "multivariate_normal": 2,
        "negative_binomial": 2,
        "noncentral_chisquare": 2,
        "noncentral_f": 3,
        "normal": 2,
        "pareto": 1,
        "poisson": 1,
        "power": 1,
        "rayleigh": 1,
        "standard_cauchy": 0,
        "standard_exponential": 0,
        "standard_gamma": 1,
        "standard_normal": 0,
        "standard_t": 1,
        "triangular": 3,
        "uniform": 2,
        "vonmises": 2,
        "wald": 2,
        "weibull": 1,
        "zipf": 1,
        "gaussian_mixture": 0,
    }

    permitted_activation_functions = [
        "cos",
        "sin",
        "exp",
        "arctan",
        "tanh",
        "ReLu",
        "Elu",
        "SoftPlus",
        "cos_and_sin",
    ]

    def __init__(self):
        pass

    @staticmethod
    def check_distribution_requirements(
            distribution: str,
            distribution_parameters: dict
    ):

        if distribution == "gaussian_mixture":
            return
        if distribution not in RandomFeaturesGeneratorr.distribution_requirements:
            raise Exception(
                f"{distribution} is not permitted. If you need it, do not be lazy and update the class"
            )
        elif (
                len(distribution_parameters)
                != RandomFeaturesGeneratorr.distribution_requirements[distribution]
        ):
            raise Exception(
                f"{distribution} "
                f"requires {RandomFeaturesGeneratorr.distribution_requirements[distribution]} basic_parameters"
            )

    @staticmethod
    def get_specific_gaussian_mixture_specifications(
            seed: int,
            gamma: list,
            activation: str,
            number_features_in_subset: int,
            increment_seed: Optional[int] = 0,
    ) -> Tuple[int, dict]:
        """
        returns specification to generate random features
        This was selected a benchmark case
        """
        spec = (
            int((seed + 1) * 1e3) + increment_seed,
            {
                "distribution": "gaussian_mixture",
                "distribution_parameters": gamma,
                "activation": activation,
                "number_features": number_features_in_subset,
                "bias_distribution": None,
                "bias_distribution_parameters": [
                    -np.pi,
                    np.pi,
                ],
            },
        )
        return spec

    @staticmethod
    def generate_random_features_from_list_with_potential_ranking(
            seed: int,
            msrr: bool,
            factor_msrr: bool,
            number_features_in_subset: int,
            pre_specified_list_of_specs_for_random_features: list,
            date_ids: np.ndarray,
            y_train: np.ndarray,
            X_train: np.ndarray
    ):
        """
        This is a key function for generating random features, with the possibility to cross-sectionally rank them
        If msrr or factor_msrr, then we will actually multiply signals * returns
        Parameters
        ----------
        seed :
        msrr :
        factor_msrr :
        number_features_in_subset :
        pre_specified_list_of_specs_for_random_features :
        date_ids :
        y_train :
        X_train :

        Returns
        -------

        """
        random_features_inside = (
            RandomFeaturesGeneratorr.generate_random_features_from_lisst(
                seed=seed,
                features=X_train,
                pre_specified_list_of_specs=pre_specified_list_of_specs_for_random_features[1],
                number_features_in_subset=number_features_in_subset
            )
        )

        ranking = pre_specified_list_of_specs_for_random_features[0]
        if ranking is not None:
            random_features_inside1 = rank_features_cross_sectionally(random_features_inside,
                                                                      date_ids,
                                                                      ranking,
                                                                      use_pandas=True)
        else:
            random_features_inside1 = random_features_inside.copy()

        if msrr:
            random_features_inside1 = build_managed_returns_for_msrr(random_features_inside1,
                                                                     y_train,
                                                                     factor_msrr,
                                                                     date_indices_for_panel=date_ids)
        # if np.abs(random_features_inside1).sum() == 0:
        #     print(f'damn zeros')
        #     breakpoint()
        return random_features_inside1

    @staticmethod
    def generate_random_features_from_lisst(
            features: np.ndarray,
            pre_specified_list_of_specs: list,
            single_precision: Optional[bool] = False,
            seed: int = None,
            number_features_in_subset: int = None,
            just_return_original_features: bool = False
    ) -> np.ndarray:
        """
        given a list of different specifications, generate random features for each of them
        normally, it should receive specifications as pre_specified_list_of_specs
        but if not, then we give specification details explicitly
        Parameters
        ----------
        number_features_in_subset: number of features per gamma
        features: actual raw features
        reproducible, yet each chuck must yield new features. increment_seed takes care of it
        single_precision: this is for memory saving and gpu
        pre_specified_list_of_specs: the main thing: a list specifying all details of random features
        seed: random seed

        Returns
        -------

        """
        if just_return_original_features:
            return features
        list_of_specs = pre_specified_list_of_specs
        # first we need to take care of seeds that are changing for every small chunk;
        # hence we need the increment_seed parameter:
        list_of_specs = [(seed, x) for x in list_of_specs]
        if number_features_in_subset:
            for x in list_of_specs:
                x[1]["number_features"] = number_features_in_subset
            # number_features_in_subset could also be defined in the algorithm and may potentially be different
            # for each block, hence we need to be able to overwrite it
        # now we build the random features
        random_features_for_generation = []
        if list_of_specs[0][0] == "mix":
            list_of_specs = list_of_specs[0][1]
        for spec in list_of_specs:
            # spec[0] is just an index to initialize the np.random.seed
            if len(spec[1]) == 4:
                random_features_for_generation.append(
                    RandomFeaturesGeneratorr.generate_random_binning_features(
                        spec[0], **spec[1]
                    )
                )
            elif len(spec[1]) == 6:
                random_features_for_generation.append(
                    (
                        RandomFeaturesGeneratorr.generate_random_neuron_features(
                            features, spec[0], **spec[1]
                        )
                    )
                )
        if len(list_of_specs) > 1:
            merged_features = np.concatenate(random_features_for_generation, axis=1)
            perm = np.random.permutation(merged_features.shape[1])
            merged_features = merged_features[:, perm]
            return merged_features

        else:
            random_features_for_generation = random_features_for_generation[0]
            if single_precision:
                random_features_for_generation = random_features_for_generation.astype(np.float32)
            return random_features_for_generation

    @staticmethod
    def apply_activation_to_multiplied_signals(
            multiplied_signals,
            activation: str
    ) -> np.ndarray:
        """
        this method takes as input signaled already multipled by some weights + cosntant: w*x+b
        and returns act(w*x+b)
        :rtype: object
        """
        if activation in ["cos", "sin", "exp", "arctan", "tanh"]:
            final_random_features = getattr(np, activation)(multiplied_signals)
        elif activation == "cos_and_sin":
            final_random_features = np.concatenate(
                [np.cos(multiplied_signals), np.sin(multiplied_signals)], axis=0
            )

        elif isinstance(activation, str) and activation.lower() == "relu":
            final_random_features = multiplied_signals * (multiplied_signals > 0)
        elif isinstance(activation, list) and activation[0].lower() == "elu":
            final_random_features = (
                                            multiplied_signals * (multiplied_signals > 0)
                                    ) + activation[1] * (np.exp(multiplied_signals) - 1) * (
                                            multiplied_signals < 0
                                    )
        elif activation.lower() == "softplus":
            final_random_features = np.log(1 + np.exp(multiplied_signals))
        elif activation.lower() == "linear":

            final_random_features = multiplied_signals
        else:
            raise Exception(f"activation function={activation} is not yet supported")
        return final_random_features

    @staticmethod
    def add_bias(
            multiplied_signals,
            bias_distribution,
            bias_distribution_parameters,
            seed=0,
    ):
        """
        Careful, multiplied signals are assumed to be P \times n where P is the sumber of signals
        and n the number of observations
        Parameters
        ----------
        multiplied_signals :
        bias_distribution :
        bias_distribution_parameters :
        Returns
        -------
        """
        np.random.seed(seed)
        number_features = multiplied_signals.shape[0]
        random_bias = getattr(np.random, bias_distribution)(
            *bias_distribution_parameters, [number_features, 1]
        )
        # here we are using numpy broadcasting to add the same bias every time period
        multiplied_signals += random_bias
        return multiplied_signals

    @staticmethod
    def generate_random_neuron_features(
            features: np.ndarray,
            random_seed: int,
            distribution: str,
            distribution_parameters: dict,
            activation: str,
            number_features: int,
            bias_distribution=None,
            bias_distribution_parameters=None,
    ) -> np.ndarray:
        """

        this function builds random neuron features f(w'S+bias) where w is
        a vector of random weights and f is an activation function, and bias is a random bias

        One important special case is a gaussian mixture, whereby we generate random scales (gamma) from the interval
        [gamma_min, gamma_max], uniformly

        :param distribution_parameters:
        :param distribution:
        :param activation:
        :param number_features:
        :param bias_distribution:
        :return:
        """
        np.random.seed(random_seed)
        signals = features

        RandomFeaturesGeneratorr.check_distribution_requirements(
            distribution, distribution_parameters
        )
        if bias_distribution:
            RandomFeaturesGeneratorr.check_distribution_requirements(
                bias_distribution,
                bias_distribution_parameters,
            )

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        if activation == "cos_and_sin":
            size = [number_signals, int(number_features / 2)]

        # first we initialize the random seed

        # X = np.random.normal(0, a) means X is distributed as Normal(0, a^2).  (a=standard deviation)
        # This is an important property of Gaussian distributions: multiplying by a constant keeps is Gaussian,
        # just scales the standard deviation
        if distribution != "gaussian_mixture":
            random_vectors = getattr(np.random, distribution)(
                *distribution_parameters, size
            )
        else:
            random_vectors = getattr(np.random, "standard_normal")(size)
            gamma_values = distribution_parameters
            minimal_gamma = gamma_values[0]
            maximal_gamma = gamma_values[1]
            all_gamma_values = np.random.uniform(
                minimal_gamma, maximal_gamma, [1, size[1]]
            )
            # now we use numpy broadcasting to do elemen-wise multiplication.
            random_vectors = random_vectors * all_gamma_values

        # w'x, where w is our random vector
        multiplied_signals = np.matmul(random_vectors.T, signals.T)
        if bias_distribution:
            multiplied_signals = RandomFeaturesGeneratorr.add_bias(
                multiplied_signals, bias_distribution, bias_distribution_parameters
            )
        final_random_features = (
            RandomFeaturesGeneratorr.apply_activation_to_multiplied_signals(
                multiplied_signals, activation
            )
        )

        return final_random_features.T

    @staticmethod
    def generate_random_binning_features(signals: np.ndarray,
                                         distribution_requirements: dict,
                                         distribution: str,
                                         distribution_parameters: list,
                                         number_features: int,
                                         random_rotation=False,
                                         random_seed=0):
        """
           WARNING: THE FEATURES ARE SUPPOSED TO BE NORMALIZED!!!
           ALWAYS PRE-PROCESS THE DATA (USING ROLLING WINDOW) !!!
           signals are assumed to be T \times M
           :param random_seed:
           :param random_rotation:
           :param distribution_parameters:
           :param distribution:
           :param distribution_requirements:
           :param signals:
           :param number_features:
           :return:
           """

        RandomFeaturesGeneratorr.check_distribution_requirements(distribution,
                                                                 distribution_parameters)

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        np.random.seed(random_seed)
        if random_rotation:
            rotate = np.random.randn(signals.shape[1], signals.shape[1])
            tmp = np.matmul(rotate, rotate.T)
            _, eigvec = np.linalg.eigh(tmp)
            # now, eigenvectors give a random rotation
            signals_rotated = np.matmul(eigvec.T, signals.T).T
        else:
            signals_rotated = signals.copy()
        delta = getattr(np.random, distribution)(*distribution_parameters, size)
        delta = delta * (np.abs(delta) > (10 ** (- 10))) + (np.abs(delta) < (10 ** (- 10))) * (10 ** (-10))  # clip
        u_ = np.random.uniform(0, 1, [number_signals, number_features]) * delta
        subtracted = signals_rotated.reshape(signals.shape[0], 1, number_signals) \
                     - u_.reshape(1, number_features, number_signals)
        subtracted_and_divided = subtracted / delta.reshape(1, number_features, number_signals)

        binned_signals = np.floor(subtracted_and_divided).reshape([signals.shape[0],
                                                                   signals.shape[1] * number_features])
        return binned_signals

    @staticmethod
    def generate_a_random_covariance_matrix_with_given_eigenvalues(
            eigenvalues: np.ndarray, seed=0
    ) -> np.ndarray:
        """

        Parameters
        ----------
        eigenvalues :

        Returns
        -------
        a random matrix with prescribed eigenvalues
        """
        eigenvalues = eigenvalues.flatten()
        size = len(eigenvalues)
        random_shocks = np.random.normal(size=[size, 2 * size])
        sigma = random_shocks @ random_shocks.T
        _, eigenvectors = np.linalg.eigh(sigma)
        # t1 = time.time()
        # eigenvectors = ortho_group.rvs(size, random_state=seed)
        # matrix = eigenvectors @ (np.diag(eigenvalues) @ eigenvectors.T)
        # t2 = time.time()
        # print(f'got the random O matrix in {t2 - t1}')
        matrix = eigenvectors @ (np.diag(eigenvalues) @ eigenvectors.T)
        return matrix

    @staticmethod
    def generate_random_deep_neuron_features(
            c_sigma: float,
            list_of_widths: list,
            X_train: pd.DataFrame,
            distribution: str,
            distribution_parameters: list,
            activation: str,
            bias_distribution=None,
            bias_distribution_parameters=None,
            random_seed=0,
    ):
        """
        this function builds random neuron features f(w'S)
        where w is a vector of random weights and
        f is an activation function
        :param random_seed:
        :param bias_distribution_parameters:
        :param distribution_parameters:
        :param distribution:
        :param activation:
        :param bias_distribution:
        :return:
        """
        np.random.seed(random_seed)
        # careful: this choice of seed will determine the sampling
        # from np.random.randint(0, 10 ** 6) below
        features = X_train.copy()
        for width in list_of_widths:
            features = np.sqrt(
                c_sigma / width
            ) * RandomFeaturesGeneratorr.generate_random_neuron_features(
                features=features,
                distribution=distribution,
                distribution_parameters=distribution_parameters,
                activation=activation,
                number_features=width,
                bias_distribution=bias_distribution,
                bias_distribution_parameters=bias_distribution_parameters,
                random_seed=np.random.randint(0, 10 ** 6),
            )

        return features


if __name__ == "__main__":
    c_sigma = 3.0
    list_of_widths = [300, 300, 600]

    X_train = np.random.normal(size=(100, 100))

    distribution = "gaussian_mixture"
    gamma = [1.0, 1.0]
    activation = "relu"

    random_features = RandomFeaturesGeneratorr.generate_random_deep_neuron_features(
        c_sigma, list_of_widths, X_train, distribution, gamma, activation
    )
    breakpoint()
