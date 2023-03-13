import time
import numpy as np
import pandas as pd
import torch
import logging
from copy import deepcopy
from typing import Optional, Tuple
from operator import itemgetter
from helpers.auxilliary_functions import sharpe_ratio, simple_markowitz

import sys

sys.path.append('../infinite_features_for_AQR')
# export PYTHONPATH="${PYTHONPATH}:/Users/malamud/Dropbox/MY_STUFF/TRADING/infinite_features_for_AQR"

import rf.RandomFeaturesGeneratorr
from helpers.auxilliary_functions import rank_features_cross_sectionally
from helpers.transaction_costs_optimization import giant_msrr_with_t_cost, smooth_positions_naively
from utils.printing import print_header

from accelerators.torch_linear_algebra import torch_eigh_from_rSVD
from rf.RandomFeaturesGeneratorr import RandomFeaturesGeneratorr
from data_preprocessing.process_futures_data import Futures

logging.basicConfig(level=logging.INFO)
if torch.cuda.is_available():
    import cupy as cp
    from utils.cp_linear_algebra import (
        cp_eigh,
        cp_three_matrices_multiplication,
        cp_tilda_S_k,
    )

    logging.info(f"Max GPU Memory Pool: {cp.get_default_memory_pool().get_limit()}b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_parameters_old():
    number_simulations = 100
    gamma_values = [0.5, 1, 2, 4]
    parameters = [{'distribution': 'normal',
                   'distribution_parameters': [0, gamma],
                   'activation': 'cos',
                   'number_features': 12000,
                   'bias_distribution': 'uniform',
                   'bias_distribution_parameters': [0, gamma]} for gamma in gamma_values] * number_simulations
    # this is a python trick to repeat stuff number_simulations times

    add_binning_features = False
    if add_binning_features:
        parameters += [{'distribution': 'normal',
                        'distribution_parameters': [0, 1],
                        'number_features': 12000,
                        'random_rotation': True}] * number_simulations \
                      + [{'distribution': 'normal',
                          'distribution_parameters': [0, 1],
                          'number_features': 12000,
                          'random_rotation': False}] * number_simulations
    ################################################################################################################
    # basic_parameters = generate_parameters_for_neural_features()  # this will be a giant thing. To be tested
    ################################################################################################################

    return parameters


def generate_parameters(number_simulations=100,
                        gamma_values=None,
                        number_features_per_gamma=500,
                        binning=False):
    if gamma_values is None:
        gamma_values = [0.1, 0.5, 1, 2, 4, 8, 16]
    if not binning:
        types = [{'distribution': 'normal',
                  'distribution_parameters': [0, gamma],
                  'activation': activation,
                  'number_features': number_features_per_gamma,
                  'bias_distribution': None,
                  'bias_distribution_parameters': [0, gamma]}
                 for gamma in gamma_values for activation in ['cos_and_sin']]
    else:
        types = [{'distribution': 'normal',
                  'distribution_parameters': [0, gamma],
                  'number_features': number_features_per_gamma,
                  'random_rotation': True} for gamma in gamma_values] \
                + [{'distribution': 'normal',
                    'distribution_parameters': [0, gamma],
                    'number_features': number_features_per_gamma,
                    'random_rotation': False} for gamma in gamma_values]
    parameters = types * number_simulations
    # this is a python trick to repeat stuff number_simulations times
    return types, parameters


def get_number_features_per_gamma(max_number_of_features, gamma_list, number_signals, use_binning, only_mix,
                                  use_constant=True):
    if use_binning:
        if only_mix:
            number_features_per_gamma = int(max_number_of_features / (len(gamma_list) * 2 * number_signals)) + 1
            # plus 1 to ensure the max_number_of_features of features is produced
        else:
            number_features_per_gamma = int(max_number_of_features / (2 * number_signals)) + 1
    else:
        if use_constant:
            number_features_per_gamma = 5000  # to adapt with the previous results
        else:
            if only_mix:
                number_features_per_gamma = int(max_number_of_features / len(gamma_list)) + 2
                # plus 2 to ensure the max_number_of_features of features is produced, not 1 because
                # if activation == 'cos_and_sin', features_size = [number_signals, int(number_features / 2)]
            else:
                number_features_per_gamma = max_number_of_features
    return number_features_per_gamma


def build_the_q_vector(
        psi_matrix: np.ndarray,
        y_train: np.ndarray,
        shrinkage_list: list,
        number_random_features: int,
        normalize_p: bool = False,
        gpu: Optional[bool] = False,
        rsvd_sample: Optional[int] = 0,
):
    """

    :param psi_matrix:
    :type psi_matrix:
    :param y_train:
    :type y_train:
    :param shrinkage_list:
    :type shrinkage_list:
    :param number_random_features:
    :type number_random_features:
    :param normalize_p:
    :type normalize_p:
    :return: WARNING:

    THE OUTPUT q_vector IS A NUMPY ARRAY FOR A REGRESSION PROBLEM (WHEN y_train HAVE ONE COLUMN)
    BUT IF IT IS A LIST FOR A CLASSIFICATION PROBLEM WHEN y_train HAVE MANY COLUMNS!
    IN THIS CASE, THE LENGTH OF THE LIST q_vector equals the number of label columns

    :rtype:
    """
    sample_size = psi_matrix.shape[0]
    covariance = psi_matrix / (
            sample_size * (number_random_features if normalize_p else 1)
    )

    # this is T \times T
    # signals.shape[0] is the number of observations
    if gpu:

        if rsvd_sample > 0:
            # These are already numpy array
            eigval, eigvec1 = torch_eigh_from_rSVD(covariance, rsvd_sample, 5, 0)
        else:
            # eigval, eigvec1 = cp_eigh(covariance)
            start = time.monotonic()
            # GPU acceleration
            covariance_gpu = torch.Tensor(covariance).to(device)
            end_copy = time.monotonic()
            move_to_gpu_time = end_copy - start
            logging.info(f"Move to GPU Time: {move_to_gpu_time:.3f}")

            eigval, eigvec1 = torch.linalg.eigh(covariance_gpu)

            end = time.monotonic()
            eig_time = end - start
            logging.info(f"Eigenvalue Time: {eig_time:.3f}")
            eigval = eigval.cpu().numpy()
            eigvec1 = eigvec1.cpu().numpy()

    else:
        # Bottleneck when high dimensional
        eigval, eigvec1 = np.linalg.eigh(covariance)

    # now we filter away low eigenvalues.
    # Why is this legal?
    # So we are using (zI + X'X)^{-1} X'y
    # we have the polar decomposition X'=HV
    # (we do not use it in the analysis, so this is just for mathematical explanations)
    # Then, X'X= HVV'H=H^2.
    # So we are doing (zI+H^2)^{-1}H Vy
    # Then, the key observation is that if H has a kernel we can ignore its action on the kernel
    # Namely, H^2 = U D U' and (zI+H^2)^{-1}H = U (z+D)^{-1} D U'
    # and we see that whatever happens on where D=0 gets annihilated
    eigvec1 = eigvec1[:, eigval > 10 ** (-10)]
    eigval = eigval[eigval > 10 ** (-10)]

    logging.info(f"Selected Eigval: {len(eigval)}")
    # now eigvec1 is a bit smaller, T \times T1 for some T1<T

    # so here we could potentially do many columns in the y_train
    # multiplied = (1 / eigval).reshape(-1, 1) * (eigvec1.T @ (covariance @ y_train))
    # this is a calculation from the paper:
    # (S'S+zI)^{-1}S'y=S'Q where
    # Q = V(D+zI)^{-1} multiplied
    # where, since covariance = V DV'
    # multiplied = D^{-1} V' covariance y=V'y
    # this vector is now T1 \times number_label_columns (if we have 1-hot encoding)
    # note however that this calculation can be significantly simplified:
    multiplied = eigvec1.T @ y_train  # number_eigenvalues \times 1

    # here it is subtle as the dimension of eigvec might be lower than that of beta !!!
    # but normalized has the right dimension !!
    if (
            len(y_train.shape) > 1 and y_train.shape[1] > 1
    ):  # then we are doing multi-class

        normalized = [
            np.concatenate(
                [
                    (1 / (eigval + z)).reshape(-1, 1)
                    * multiplied[:, i].reshape(-1, 1)
                    for z in shrinkage_list
                ],
                axis=1,
            )
            for i in range(multiplied.shape[1])
        ]

        q_vector = [eigvec1 @ normalized[i] for i in range(multiplied.shape[1])]

    else:

        # this is (T \times T1) \times (T1 \times len(shrinkage))
        # which should give T \times len(shrinkage)

        normalized = np.concatenate(
            [
                (1 / (eigval + z)).reshape(-1, 1) * multiplied.reshape(-1, 1)
                for z in shrinkage_list
            ],
            axis=1,
        )
        # eignevc1 is T \times num_eigenvalues
        q_vector = [eigvec1 @ normalized]

    len_eigenval = len(eigval.tolist())
    if len_eigenval < number_random_features:
        # the true psi matrix is P \times P/ If P>T, then it will have zeros
        eigval = np.array(
            eigval.tolist() + [0] * (number_random_features - len_eigenval)
        )
    else:
        # otherwise the first number_random_features - len(psi_hat_eig.tolist())
        # eigenvalues are identically zero
        eigval = eigval[(number_random_features - len_eigenval):]
    return q_vector, eigval


def get_block_sizes(
        number_random_features: int, small_subset_size: int, voc_grid: list
) -> list:
    """returns a list of block sizes"""
    block_sizes = (
        np.arange(0, number_random_features, small_subset_size).astype(int).tolist()
    )
    if number_random_features not in block_sizes:
        block_sizes += [number_random_features]

    # if grid point in voc_grid is not in block_sizes, we add them in the block_sizes
    block_sizes = list(set(block_sizes + voc_grid))

    block_sizes.sort()  # sort grid points
    return block_sizes


def compute_psi_matrix_and_q_vectors_for_voc_grid(
        seed: int,
        block_sizes: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shrinkage_list: list,
        voc_grid: list,
        test_mode: bool,
        produce_voc_curve: bool,
        normalize_p: bool,
        gpu: Optional[bool] = False,
        rsvd_sample: Optional[int] = 0,
        msrr: bool = False,
        factor_msrr: bool = False,
        date_indices_for_panel: list = None,
        pre_specified_list_of_specs_for_random_features: list = None,
):
    """

    :param seed:
    :param block_sizes:
    :param X_train:
    :param y_train:
    :param shrinkage_list:
    :param voc_grid:
    :param test_mode:
    :param produce_voc_curve:
    :param normalize_p:
    :param gpu:
    :param rsvd_sample:
    :param msrr: This is the key new variable. If True, then we are actually trading
    an efficient portfolio of (R * randomFeatures)
    :return:
    """
    np.random.seed(seed)

    psi_matrix = None

    psi_eigenvalues_for_expanding_complexity = dict()
    q_vectors_for_expanding_complexity = dict()
    sigma_hat_eigenvalues_for_expanding_complexity = dict()

    random_features_all = []
    k = 0
    total_q_vectors = 0
    print_header("Computing Psi Matrix and Q vector")
    for block in range(len(block_sizes) - 1):
        if block % 10 == 0:
            logging.info(f"Block {block}/{len(block_sizes)}")
        k += 1
        # now we loop through blocks of features
        number_features_in_subset = block_sizes[block + 1] - block_sizes[block]
        random_features = RandomFeaturesGeneratorr.generate_random_features_from_list_with_potential_ranking(
            seed=int((seed + 1) * 1e3) + k,
            msrr=msrr,
            factor_msrr=factor_msrr,
            number_features_in_subset=number_features_in_subset,
            pre_specified_list_of_specs_for_random_features=pre_specified_list_of_specs_for_random_features,
            date_ids=date_indices_for_panel,
            y_train=y_train,
            X_train=X_train
        )

        if test_mode:
            random_features_all.append(random_features)

        psi_matrix = update_the_psi_matrix(gpu,
                                           psi_matrix,
                                           random_features)

        if produce_voc_curve and (block_sizes[block + 1] in voc_grid):
            logging.info(f"Total Q Vectors:\t{total_q_vectors + 1}/{len(voc_grid)}")
            # so now we are running the regression on the intermediate
            # result with a subset of random features
            start = time.monotonic()
            y_train_to_use = np.ones([psi_matrix.shape[0], 1]) if msrr else y_train
            q_vector, psi_hat_eig = build_the_q_vector(
                psi_matrix,
                y_train_to_use,  # if msrr then we are regressing 1 on managed returns
                shrinkage_list,
                number_random_features=block_sizes[block + 1],
                normalize_p=normalize_p,
                gpu=gpu,
                rsvd_sample=rsvd_sample,
            )

            q_vectors_for_expanding_complexity.update(
                {block_sizes[block + 1]: q_vector}
            )

            psi_eigenvalues_for_expanding_complexity.update(
                {block_sizes[block + 1]: psi_hat_eig}
            )
            end = time.monotonic()
            vector_time = end - start
            logging.info(f"Build The q vector: {vector_time:.3f}s")
            total_q_vectors += 1

    if test_mode:
        random_features = np.concatenate(random_features_all, axis=1)
        # Covariance matrix
        true_psi_matr = random_features.T @ random_features
    else:
        true_psi_matr = None
        random_features = None
    if produce_voc_curve:
        voc_curve = {
            "psi_eig": psi_eigenvalues_for_expanding_complexity,
            "q_vectors": q_vectors_for_expanding_complexity,
            "sigma_eig": sigma_hat_eigenvalues_for_expanding_complexity,
        }
    else:
        voc_curve = dict()
    return psi_matrix, true_psi_matr, random_features, voc_curve


def update_predictions(predictions: dict,
                       t_or_t: str,
                       random_features: dict,
                       beta_chunks: dict,
                       normalize_p: bool,
                       voc_curve: dict,
                       block_sizes: list,
                       block: int,
                       number_labels: int
                       ):
    """

    Parameters
    ----------
    predictions : dictionary with keys (bock, column number). column number only matters if we are
    solving a multi-class classification problem
    t_or_t : 'test' or 'train'

    Returns
    -------

    """
    predictions[t_or_t].update(
        {
            (key, i): predictions[t_or_t][key, i]
                      + random_features[t_or_t]
                      @ beta_chunks[key, i]
                      / (np.sqrt(key) if normalize_p else 1)
            for key in voc_curve["q_vectors"]
            if key >= block_sizes[block + 1]
            for i in range(number_labels)
        }
    )


def initialize_predictions(data: np.ndarray,
                           shrinkage_list: list,
                           number_labels: int,
                           voc_curve: dict,
                           factor_msrr: bool,
                           dimension: int
                           ):
    zeros_init = dimension if factor_msrr else data.shape[0]

    predictions = {
        (key, i): np.zeros([zeros_init, len(shrinkage_list)])
        for key in voc_curve["q_vectors"].keys()
        for i in range(number_labels)
    }
    return predictions


def compute_betas_and_predictions(
        seed: int,
        shrinkage_list: list,
        test_and_train: dict,
        block_sizes: list,
        y_train: np.ndarray,
        y_test: np.ndarray,
        voc_curve: dict,
        produce_betas: bool,
        test: bool,
        normalize_p: bool,
        msrr: bool = False,
        factor_msrr: bool = False,
        date_indices_for_panel_dict: dict = {},
        pre_specified_list_of_specs_for_random_features: list = None
) -> Tuple[dict, dict, list]:
    """
    In our novel approach, to deal with high dimensionality, we first pre-compute psi_matrix and forget
    all random_features due to memory constraints.
    And only then, we re-compute random features again to get betas and predictions

    :param seed:
    :type seed:
    :param shrinkage_list:
    :type shrinkage_list:
    :param test_and_train:
    :type test_and_train:
    :param block_sizes:
    :type block_sizes:
    :param y_train:
    :type y_train:
    :param y_test:
    :type y_test:
    :param voc_curve:
    :type voc_curve:
    :param produce_betas:
    :type produce_betas:
    :param test:
    :type test:
    :param normalize_p:
    :type normalize_p:
    :param msrr:
    :type msrr:
    :param factor_msrr:
    :type factor_msrr:
    :param date_indices_for_panel_dict:
    :type date_indices_for_panel_dict:
    :return:
    :rtype:
    """
    # here it is very important that we re-build the same seeds !!!!
    np.random.seed(seed)

    try:
        number_labels = y_train.shape[1]
    except:
        number_labels = 1
        logging.info(f"Number of Labels: 1")
    # I am afraid to single out the next loop into a function so that the seed is not lost
    # first we initialize the output with empty lists and zeros
    betas = {
        (key, i): [] for key in voc_curve["q_vectors"] for i in range(number_labels)
    }
    # realized_in_sample_returns = {
    #     (key, i): np.zeros([1, len(shrinkage_list)])
    #     for key in voc_curve["q_vectors"].keys()
    #     for i in range(number_labels)
    # }

    predictions = {key: initialize_predictions(data=test_and_train[key],
                                               shrinkage_list=shrinkage_list,
                                               number_labels=number_labels,
                                               voc_curve=voc_curve,
                                               factor_msrr=factor_msrr,
                                               dimension=len(np.unique(date_indices_for_panel_dict['test']))
                                               )
                   for key in test_and_train
                   }

    future_random_features_all = list()
    future_random_features_params = []
    k = 0
    for block in range(len(block_sizes) - 1):
        k += 1
        # logging.info(f"Predictions for block: {k}/{len(block_sizes) - 1}")
        # now we loop through blocks of features
        number_features_in_subset = block_sizes[block + 1] - block_sizes[block]

        ys = {'test': y_test, 'train': y_train}

        random_features = {key: RandomFeaturesGeneratorr.generate_random_features_from_list_with_potential_ranking(
            seed=int((seed + 1) * 1e3) + k,
            msrr=msrr,
            factor_msrr=factor_msrr,
            number_features_in_subset=number_features_in_subset,
            pre_specified_list_of_specs_for_random_features=pre_specified_list_of_specs_for_random_features,
            date_ids=date_indices_for_panel_dict[key],
            y_train=ys[key],
            X_train=np.nan_to_num(test_and_train[key])
        ) for key in test_and_train}

        if test:
            future_random_features_all.append(random_features['test'])
        # q_vector is T \times len(shrinkage_list)
        # random_features is T \times P1
        # hence beta_chunk \in \R^{P_1\times len(shrinkage_list)}
        # so the betas for the chunk will only matter for a model with hih enough complexity
        # hence the condition key >= block_sizes[block + 1]

        start = time.monotonic()

        # below we are dividing betas by T = test_and_train['train'].shape[0]
        # this is not a glitch: it is necessary because we need to do (z+S'S/T)^{-1}S'y/T

        beta_chunks = {
            (key, i): (
                    random_features['train'].T
                    @ voc_curve["q_vectors"][key][i]
                    / (test_and_train['train'].shape[0] * np.sqrt(key if normalize_p else 1))
            )
            for key in voc_curve["q_vectors"]
            if key >= block_sizes[block + 1]
            for i in range(number_labels)
        }
        end = time.monotonic()
        chunks_time = end - start
        # logging.info(f"Beta Chunks time: {chunks_time:.3f}s")

        start = time.monotonic()
        update_predictions(predictions,
                           t_or_t='test',
                           random_features=random_features,
                           beta_chunks=beta_chunks,
                           normalize_p=normalize_p,
                           voc_curve=voc_curve,
                           block_sizes=block_sizes,
                           block=block,
                           number_labels=number_labels,
                           )
        end = time.monotonic()
        future_time = end - start
        # logging.info(f"future predictions time: {future_time:.3f}s")
        # logging.info(f"In sample time: {in_sample_time:.3f}s")
        # same here: only stuff with high complexity,
        # if key >= block_sizes[block + 1], gets updated

        # realized_in_sample_returns.update(
        #     {
        #         (key, i): realized_in_sample_returns[key]
        #         + (
        #             beta_chunks[key, i].T
        #             @ random_features.T
        #             @ y_train
        #             / np.sqrt(key if normalize_p else 1)
        #         ).T
        #         for key in realized_in_sample_returns
        #         if key >= block_sizes[block + 1]
        #         for i in range(number_labels)
        #     }
        # )

        # so the amazing thing is that we do not need to actually store the betas.
        # we update predictions chunk-by-chunk and can forget them
        if produce_betas:
            betas.update(
                {
                    (key, i): betas[key, i] + [beta_chunks[key, i]]
                    for key in voc_curve["q_vectors"]
                    if key >= block_sizes[block + 1]
                    for i in range(number_labels)
                }
            )
        # now we have a problem that some raw features are nans
        # predictions always have number of columns equal to the len(shrinkage_list)
        # we now kill all predictions for which raw features had even a single nan
        # for tt in test_and_train:
        """
        # so for training, this is obvious that we need to be careful 
        # and not just blindly train on nans replaced with zeros
        # for testing, however, we should think, it depends on the method. 
        # if type ==  'giant_panel_regression', the trade is R * f(S), and since it was trained of non-missing S, 
        computing f(S). Well, then, kind of the same logic applies to any other implementations. 

        """
        # predictions[tt].update({key: predictions[tt][key] * (np.isnan(test_and_train[tt]).sum(1) == 0).reshape(-1, 1)
        #                         for key in predictions[tt]})
    # realized_in_sample_returns = {
    #     (key, i): realized_in_sample_returns[key, i] / y_train.shape[0]
    #     for key in realized_in_sample_returns
    #     for i in range(number_labels)
    # }

    # here we divide by T, because the estimator of b_star_hat_in_sample
    # is designed to take stuff normalized by T
    if produce_betas:
        betas = {
            (key, i): np.concatenate(betas[key, i], axis=0)
            for key in voc_curve["q_vectors"]
            for i in range(number_labels)
        }

    return (
        betas,
        predictions,
        future_random_features_all,
    )


def compute_strategy_returns_for_giant_msrr(
        tickers: list,
        number_features_per_ticker: int,
        q_vector: np.ndarray,
        seed: int,
        slice_: dict,
        normalize_p: bool = False,
        pre_specified_list_of_specs_for_random_features: list = None,
        seed_step: int = 1e3,
        shrinkage_list: list = None,
        use_original_signals: bool = False,
        return_betas: bool = False
) -> tuple:
    """"""

    sig_slices = {'test': slice_['test_sigs'],
                  'train': slice_['train_sigs']}

    ret_slices = {'test': slice_['test_ret'],
                  'train': slice_['train_ret']}
    beta_chunk_list = list()
    oos_msrr_returns = list()

    strategy_returns = np.zeros([ret_slices['test'].shape[0], len(shrinkage_list)])
    for ii, ticker in enumerate(tickers):
        # managed returns, ins and oos
        random_features = {key:
                               RandomFeaturesGeneratorr.generate_random_features_from_lisst(
                                   seed=int((seed + 1) * seed_step) + ii,
                                   features=sig_slices[key][ticker].values,
                                   pre_specified_list_of_specs=pre_specified_list_of_specs_for_random_features,
                                   number_features_in_subset=number_features_per_ticker,
                                   just_return_original_features=use_original_signals
                               ) * ret_slices[key].values[:, ii].reshape(-1, 1)
                           for key in ['test', 'train']
                           }
        beta_chunks = random_features['train'].T @ q_vector[0] \
                      / (sig_slices['train'][ticker].shape[0]
                         * np.sqrt(number_features_per_ticker if normalize_p else 1))
        if return_betas:
            beta_chunk_list += [beta_chunks]
            oos_msrr_returns += [random_features['test']]
        # so, in fact beta chunks are the actual portfolio weights of the managed portfolio
        strategy_returns += random_features['test'] @ beta_chunks / (
            np.sqrt(number_features_per_ticker) if normalize_p else 1)
    breakpoint()
    strategy_returns = pd.DataFrame(strategy_returns,
                                    index=ret_slices['test'].index,
                                    columns=shrinkage_list)
    return strategy_returns, beta_chunk_list


def transform_predictions_into_strategy_returns(predictions: np.ndarray,
                                                asset_returns: np.ndarray,
                                                date_ids: np.ndarray,
                                                shrinkage_list: list,
                                                msrr: bool = False,
                                                factor_msrr: bool = False) -> dict:
    """
    predictions are flattened. Each column predictions[:, i] in fact corresponds to a panel of predictions,
    and column i corresponds to a particular shrinkage method.

    Parameters
    ----------
    factor_msrr
    msrr
    shrinkage_list
    date_ids
    asset_returns
    predictions :

    Returns
    -------

    """
    strategy_returns = pd.DataFrame(predictions, columns=shrinkage_list)
    if msrr:
        # in this case, we were running a panel msrr, and we still need to sum over assets
        # but no need to multiply by asset returns because predictions are returns
        if factor_msrr:
            # these are already factors, summed across assets
            strategy_returns.set_index(np.unique(date_ids), inplace=True)
        else:
            strategy_returns = strategy_returns.set_index(date_ids).groupby(date_ids).sum()
    else:
        # this means we have run a panel regression, predicting actual asset returns
        strategy_returns = (strategy_returns * asset_returns.reshape(-1, 1)).groupby(date_ids).sum()
    return strategy_returns


def ridge_regression_with_giant_number_of_random_features(
        X_train: np.ndarray,
        y_train: np.ndarray,
        shrinkage_list: list,
        number_random_features: int,  # P
        small_subset_size: int,  # P'
        seed: int,
        X_test: np.ndarray,
        y_test: np.ndarray,
        voc_grid: list,
        test_mode: Optional[bool] = False,
        produce_voc_curve: Optional[bool] = False,
        produce_betas: Optional[bool] = False,
        run_linear_model: Optional[bool] = False,
        normalize_p: Optional[bool] = False,
        gpu: Optional[bool] = False,
        rsvd_sample: Optional[int] = 0,
        msrr: bool = False,
        factor_msrr: bool = False,
        date_indices_for_panel_dict: dict = {},
        parameters_for_random_features: list = None,
) -> dict:
    """
    Important: the code assumes that stock ids are already sorted!
    Pre-process the data so that stock ids are increasing !!
    so the original data must first be sorted on dates.
    Then, conditional on any date we sort stock ids.
    And this pre-sorted data comes into the function

    Same for date_ids: We are appending data!
    So we assume that

    Parameters
    ----------

    date_indices_for_panel_dict: dictionary with 'test' and 'train', showing dates for train and test slices.
    Is important for factor construction. And is also important if we need to do cross-sectional ranking on each date
    factor_msrr: if True, then we build factors \sum_i R_i S_i(k), and then do MSRR on them
    msrr; If True, then we do msrr on R*S
    rsvd_sample: this is if we need smart algorithms for eigenvalue decomposition
    gpu: If True, then we use GPU for matrix multiplications
    normalize_p: if true, then we normalize features by sqrt(their number). Should be set to false
    run_linear_model: this parameter is for testing. If True, we verify if our method gives the same results as
    a plain linear ridge
    voc_grid: grid for producing VOC curve. Must be multiples of small_subset_size
    produce_betas : If True, then we also output the giant beta vector.
    It could be huge (size = number_random_features, which could be a million or so)
    produce_voc_curve : If True, then we actually output predictions for a giant grid of numbers of random features
    (with a step size of roughly number_random_features / small_subset_size)
    test_mode : If True, then we run a test to see if the output coincides with a simple, naive linear ridge
    X_test : the chunk of out-of-sample (test) data on which we produce OOS predictions
    shrinkage_list : list of ridge shrinkage basic_parameters
    X_train : in-sample raw signals from which random features are constructed
    y_train : in-sample returns to be predicted
    number_random_features : how many random features we want to produce. Could be a very large number
    small_subset_size : we split random features into sub-groups so that they fit in memory and
    running it becomes feasible even on a small machine
    seed : random seed. One should run this for a fixed seed, and then average predictions across seeds

    Returns dictionary
    output = {'rmt_stuff': rmt_stuff,
              'betas': betas,
              'future_predictions': future_predictions}

    -------
    rmt_stuff:  random matrix theory stuff that can be used to compute the
                optimal shrinkage parameter z_*
    'betas':    actual regression betas (for each shrinkage level)
    'future_predictions':   Actual predictions for each (date-stock)
    Each of these is itself a dictionary, indexed by a grid of "numbers of random features"
    If produce_voc_curve = True, then this is an actual grid.
    If produce_voc_curve = False, then this is just one point, = number_random_features
    For each number_of_features, the corresponding future_predictions[number_of_features]
    is a matrix, dimension (OOS sample size) \times (shrinkage_list), so that for each value of
    the shrinkage parameter we have one prediction.

    Similarly for the rmt_stuff and betas.

    Why would the OOS sample size be big? Well, we do not need to
    re-compute betas every period.
    It is enough to do it every few month (say, every 3 months),
    in which case OOS sample size = number_of_stocks \times 3


    """
    t_first = time.time()
    logging.info(f"GPU: {gpu}")
    random_features_start = time.monotonic()
    sample_size = X_train.shape[0]

    block_sizes = get_block_sizes(
        number_random_features, small_subset_size, voc_grid
    )
    logging.info(f"# Block: {len(block_sizes)}")
    start_psi_matrix = time.monotonic()
    (
        psi_matrix,
        true_psi_matr,
        random_features,
        voc_curve,
    ) = compute_psi_matrix_and_q_vectors_for_voc_grid(
        seed=seed,
        block_sizes=block_sizes,
        X_train=X_train,
        y_train=y_train,
        shrinkage_list=shrinkage_list,
        test_mode=test_mode,
        produce_voc_curve=produce_voc_curve,
        voc_grid=voc_grid,
        normalize_p=normalize_p,
        gpu=gpu,
        rsvd_sample=rsvd_sample,
        msrr=msrr,
        factor_msrr=factor_msrr,
        date_indices_for_panel=date_indices_for_panel_dict['train'],
        pre_specified_list_of_specs_for_random_features=parameters_for_random_features
    )

    end_psi_matrix = time.monotonic()
    logging.info(
        f"Psi time: {(end_psi_matrix - start_psi_matrix):.3f}s \t RF: {number_random_features} \t P': {small_subset_size}"
    )
    if not produce_voc_curve:
        # q_vector \in R^{T\times len(shrinkage_list)}
        # but psi_hat_eig have lots of missing zeros. We should add them
        y_train_to_use = np.ones([psi_matrix.shape[0], 1]) if msrr else y_train
        q_vector, psi_hat_eig = build_the_q_vector(
            psi_matrix,
            y_train_to_use,
            shrinkage_list,
            number_random_features,
            normalize_p=normalize_p,
        )
        voc_curve["psi_eig"] = {number_random_features: psi_hat_eig}
        voc_curve["q_vectors"] = {number_random_features: q_vector}

    (
        betas,
        predictions,
        future_random_features_all,
    ) = compute_betas_and_predictions(
        seed=seed,
        shrinkage_list=shrinkage_list,
        test_and_train={'test': X_test, 'train': X_train},
        block_sizes=block_sizes,
        y_train=y_train,
        y_test=y_test,
        voc_curve=voc_curve,
        produce_betas=produce_betas,
        test=test_mode,
        normalize_p=normalize_p,
        msrr=msrr,
        factor_msrr=factor_msrr,
        date_indices_for_panel_dict=date_indices_for_panel_dict,
        pre_specified_list_of_specs_for_random_features=parameters_for_random_features
    )
    strategy_returns = {key: transform_predictions_into_strategy_returns(predictions=predictions['test'][key],
                                                                         asset_returns=y_test,
                                                                         date_ids=date_indices_for_panel_dict['test'],
                                                                         shrinkage_list=shrinkage_list,
                                                                         msrr=msrr,
                                                                         factor_msrr=factor_msrr)
                        for key in predictions['test']}

    random_features_end = time.monotonic()
    # b_star_hat_in_sample has now same length as shrinkage_list
    random_features_time = random_features_end - random_features_start
    output = {
        "betas": betas,
        "strategy_returns": strategy_returns,
        "random_features_time": random_features_time,
    }
    t_last = time.time()
    print(f'-----------------------------full function took {t_last - t_first}---------------------------------')
    if test_mode:
        test_giant_to_compare_with_standard_ridge(X_train,
                                                  y_train,
                                                  shrinkage_list,
                                                  output,
                                                  future_random_features_all,
                                                  number_random_features,
                                                  true_psi_matr,
                                                  sample_size,
                                                  random_features,
                                                  )

    return output


def test_giant_to_compare_with_standard_ridge(X_train,
                                              y_train,
                                              shrinkage_list,
                                              output,
                                              future_random_features_all,
                                              number_random_features,
                                              true_psi_matr,
                                              sample_size,
                                              random_features):

    timing_ridge_start = time.monotonic()
    future_random_features_all = np.concatenate(
        future_random_features_all, axis=1
    )
    beta_true = (
            np.concatenate(
                [
                    np.linalg.inv(
                        z * np.eye(number_random_features)
                        + true_psi_matr / sample_size / number_random_features
                    )
                    @ (random_features.T / np.sqrt(number_random_features))
                    @ y_train
                    for z in shrinkage_list
                ],
                axis=1,
            )
            / X_train.shape[0]
    )

    future_predictions_true = (
            future_random_features_all @ beta_true / np.sqrt(number_random_features)
    )

    timing_ridge_end = time.monotonic()
    ridge_time = timing_ridge_end - timing_ridge_start
    # logging.info(
    #     f"Please enjoy the power of math: \n"
    #     f"{betas}\n versus \n "
    #     f"{beta_true}"
    # )
    # logging.info(
    #     f"and predictions:\n"
    #     f'{output["future_predictions"]}\n'
    #     f"and "
    #     f"{future_predictions_true}"
    # )
    output["beta_true"] = beta_true
    output["random_features"] = random_features
    output["future_random_features_all"] = future_random_features_all
    output["ridge_time"] = ridge_time
    output["ridge_predictions"] = future_predictions_true


def populate_q_vectors_dictionary(produce_voc_curve: bool,
                                  block_sizes: list,
                                  voc_grid: list,
                                  is_multiclass: bool,
                                  block: int,
                                  voc_curve: dict,
                                  q_vector: None):
    if produce_voc_curve and (block_sizes[block + 1] in voc_grid):
        if is_multiclass:
            voc_curve["q_vectors"].update(
                {block_sizes[block + 1]: deepcopy(q_vector)}
            )
        else:
            voc_curve["q_vectors"].update(
                {block_sizes[block + 1]: [deepcopy(q_vector)]}
            )


def eigen_decompositions_with_top_and_not_too_small_eigenvalues(features: np.ndarray,
                                                                min_threshold: float = 10 ** (-10),
                                                                number_top_eigenvalues: int = 2000,
                                                                gpu: bool = False):
    """
    So, our goal is to get the top eigenvectors and eigenvalues of AA', where A=features
    So we use smart algebra:
    A'Av = lambda v implies
    AA' (Av) = lambda (Av)
    and \|Av\|^2=v'A'Av=lambda. Thus,
    if we can eigen-decompose A'A"
    A'A = V D V'
    we have AA' = U (D)U'
    where U = [Av D^{-1/2}] are the eigenvalues of A'A
    Parameters
    ----------
    features :
    min_threshold :
    number_top_eigenvalues :
    gpu :

    Returns
    -------

    """
    p1 = features.shape[1]
    t_ = features.shape[0]

    if t_ > p1:
        features = features.T
        large_covariance = True
    else:
        large_covariance = False

    if gpu:
        tilda_D, tilda_V = cp_eigh(features @ features.T)
    else:
        tilda_D, tilda_V = np.linalg.eigh(features @ features.T)

    # now we get rid of redundant eigenvalues. This must be done before we proceed to matrix multiplication,
    # to save dimensions
    tilda_V = tilda_V[:, tilda_D > min_threshold]
    tilda_D = tilda_D[tilda_D > min_threshold]
    tilda_V = tilda_V[:, -min(number_top_eigenvalues, min(p1, t_)):]
    tilda_D = tilda_D[-min(number_top_eigenvalues, min(p1, t_)):]

    # now, eigenvalues do not change, but eigenvectors, we need to be fixed:
    if large_covariance:
        tilda_V = features.T @ (tilda_V * (tilda_D ** (-1 / 2)).reshape(1, -1))
    return tilda_V, tilda_D


def produce_the_first_q_vector_for_zero_block(
        gpu,
        random_features,
        niu,
        p_1,
        debug,
        is_multiclass,
        y_train,
        shrinkage_list,
        produce_voc_curve,
        block_sizes,
        voc_grid,
        block,
        voc_curve,
        Vs,
        eigens,
):
    # \tilda(D), \tilda(V) = eigen(S_k'S_k)
    V_0, tilda_D = eigen_decompositions_with_top_and_not_too_small_eigenvalues(
        features=random_features,
        min_threshold=10 ** (-10),
        number_top_eigenvalues=niu,
        gpu=gpu)

    if debug:
        logging.info(f"Selected Eigenvalues: {len(tilda_D)}")
    # dictionaries
    # here as well we are using that B@A=(diag(B)) * A for diagonal matrices
    if is_multiclass:
        q_vector = [
            np.concatenate(
                [
                    V_0
                    @ (
                            (1 / (tilda_D + z)).reshape(-1, 1)
                            * (V_0.T @ y_train[:, i].reshape(-1, 1))
                    )
                    for z in shrinkage_list
                ],
                axis=1,
            )
            for i in range(y_train.shape[1])
        ]
    else:
        y_train = y_train.reshape(-1, 1)
        q_vector = np.concatenate(
            [
                V_0 @ ((1 / (tilda_D + z)).reshape(-1, 1) * (V_0.T @ y_train))
                for z in shrinkage_list
            ],
            axis=1,
        )
    populate_q_vectors_dictionary(
        produce_voc_curve,
        block_sizes,
        voc_grid,
        is_multiclass,
        block,
        voc_curve,
        q_vector,
    )

    Vs[block] = V_0
    eigens[block] = tilda_D
    return q_vector


def mathematical_q_vector_computation(
        gpu, is_multiclass, V_k, diagonals, V_k_T_y_train
):
    if gpu:
        if is_multiclass:
            q_vector = [
                np.concatenate(
                    [
                        cp_three_matrices_multiplication(
                            V_k,
                            diagonal,
                            V_k_T_y_train[:, i].reshape(-1, 1),
                            use_diagonal=True,
                        )
                        for diagonal in diagonals
                    ],
                    axis=1,
                )
                for i in range(V_k_T_y_train.shape[1])
            ]
        else:
            q_vector = np.concatenate(
                [
                    cp_three_matrices_multiplication(
                        V_k, diagonal, V_k_T_y_train, use_diagonal=True
                    )
                    for diagonal in diagonals
                ],
                axis=1,
            )
    else:

        if is_multiclass:
            # V_k has \nu columns, T rows; diagonal is \nu \times \nu
            q_vector = [
                np.concatenate(
                    [
                        V_k
                        @ (
                                np.diag(diagonal).reshape(-1, 1)
                                * V_k_T_y_train[:, i].reshape(-1, 1)
                        )
                        for diagonal in diagonals
                    ],
                    axis=1,
                )
                for i in range(V_k_T_y_train.shape[1])
            ]
        else:
            q_vector = np.concatenate(
                [
                    V_k
                    @ (
                            np.diag(diagonal).reshape(-1, 1)
                            * V_k_T_y_train.reshape(-1, 1)
                    )
                    for diagonal in diagonals
                ],
                axis=1,
            )

    # q_vector = np.concatenate(
    #     [
    #         V_k @ np.diag(1 / (lambda_k + z)) @ V_k.T @ y_train
    #         for z in shrinkage_list
    #     ],
    #     axis=1,
    # )
    return q_vector


def produce_q_vector_for_nonzero_block(
        gpu,
        sample_size,
        random_features,
        niu,
        p_1,
        debug,
        is_multiclass,
        y_train,
        shrinkage_list,
        produce_voc_curve,
        block_sizes,
        voc_grid,
        block,
        voc_curve,
        Vs,
        eigens,
):
    start_full_block = time.monotonic()
    # should be T x min(niu, P)
    previous_V = Vs[block - 1]
    previous_d = eigens[block - 1]

    # Define D_{k-1} = diag(d_{k-1})
    previous_D = np.diag(previous_d)

    # **************************
    # Xi matrix computation and
    # decomposition takes < 2s
    # For N=42k, nu=600
    # **************************

    # \Xi_k = D_{k-1} + V'_{k-1} S_k S'_k V_{k-1}
    # V is T \times min(T,(nu + p1)). So it can be T\times T for small T
    start_xi = time.monotonic()

    pre_multiplied = random_features.T @ previous_V
    Xi = previous_D + pre_multiplied.T @ pre_multiplied

    end_xi = time.monotonic()
    xi_time = end_xi - start_xi
    logging.info(f"Xi time: {xi_time:.3f}s")

    if gpu:
        eigenval_xi, eigenvec_xi = cp_eigh(Xi)
    else:
        start = time.monotonic()
        eigenval_xi, eigenvec_xi = np.linalg.eigh(Xi)
        end = time.monotonic()
        eig_xi_time = end - start
        logging.info(f"Xi decomposition time: {eig_xi_time:.3f}s")

    # **************************
    # Tilda_V_K takes < 0.2s
    # **************************
    start = time.monotonic()
    # \tilda(V_k) = V_{k-1} @ V_k^{\Xi}
    tilda_V_k = previous_V @ eigenvec_xi
    end = time.monotonic()
    tilda_v_k_time = end - start
    logging.info(f"tilda_V_k time: {tilda_v_k_time:.3f}s")

    # *********************
    # Tilda_S_K takes ~ 40s
    # ********************
    start = time.monotonic()
    # \tilda(S_k) = (I - V_{k-1}V_{k-1}')S_k

    if gpu:
        tilda_S_k = cp_tilda_S_k(sample_size, previous_V, random_features)
    else:
        # random_features are T \times P_1;
        # previous_V is T \times \nu
        tilda_S_k = random_features - previous_V @ (previous_V.T @ random_features)
        # tilda_S_k = (
        #     np.eye(sample_size) - previous_V @ previous_V.T
        # ) @ random_features
    end = time.monotonic()
    tilda_S_k_time = end - start
    logging.info(f"tilda_S_k time: {tilda_S_k_time:.3f}s")

    start = time.monotonic()
    # \Gamma_k = \tilda(S_k)' \tilda(S_k)
    gamma_k = tilda_S_k.T @ tilda_S_k
    if debug:
        assert gamma_k.shape == (
            p_1,
            p_1,
        )
    end = time.monotonic()
    gamma_k_time = end - start
    logging.info(f"gamma_k time: {gamma_k_time:.3f}")

    start = time.monotonic()
    if gpu:
        delta_k, W_k = cp_eigh(gamma_k)
    else:
        delta_k, W_k = np.linalg.eigh(gamma_k)
    delta_k = np.maximum(delta_k, 1e-19)
    # \tilda(W)_k = \tilda(S)_k @ W_k @ diag(delta_k ^ (-1/2))
    tilda_W_k = tilda_S_k @ (W_k * (1 / np.sqrt(delta_k)))

    # assert tilda_W_k.shape == (sample_size, p_1)
    end = time.monotonic()
    delta_k_tilda_W_k_time = end - start
    logging.info(f"delta_k, tilda_W_k time: {delta_k_tilda_W_k_time:.3f}s")
    hat_V_k = np.concatenate([tilda_V_k, tilda_W_k], axis=1)
    hat_lambda_k = np.concatenate([eigenval_xi, delta_k])

    # ********************
    # Sorting takes < 0.005s
    # ********************
    start_sort = time.monotonic()
    # sort values of hat_lambda_k and hat_V_k
    row_wise_hat_V_k = hat_V_k.T
    to_sort = [
        (eigenvalue, eigenvector)
        for (eigenvalue, eigenvector) in zip(hat_lambda_k, row_wise_hat_V_k)
    ]

    # sort on eigenvalues
    sorted_eigens = sorted(to_sort, key=itemgetter(0))
    end_sort = time.monotonic()
    sorting_time = end_sort - start_sort
    logging.info(f"Sorting time: {sorting_time:.3f}")

    hat_V_k = np.vstack([x[1].T for x in sorted_eigens]).T
    hat_lambda_k = np.array([x[0] for x in sorted_eigens])

    # Induction Step
    # take top niu

    lambda_k = hat_lambda_k[-niu:]
    V_k = hat_V_k[:, -niu:]
    V_k = V_k[:, lambda_k > 10 ** (-10)]
    lambda_k = lambda_k[lambda_k > 10 ** (-10)]

    eigens[block] = lambda_k
    Vs[block] = V_k
    if True:
        del eigens[block - 1]
        del Vs[block - 1]

    if debug:
        logging.info(f"Selected Eigval: {len(lambda_k)}")

    # ******************
    # Q- Vector computation,
    # can take 10s for sample_size=42k and nu=600
    # ******************
    start = time.monotonic()
    # matrix multiplication is associative, i.e. (AB)C = A(BC)
    diagonals = [np.diag(1 / (lambda_k + z)) for z in shrinkage_list]
    if not is_multiclass:
        y_train = y_train.reshape(-1, 1)
    V_k_T_y_train = V_k.T @ y_train

    q_vector = mathematical_q_vector_computation(
        gpu, is_multiclass, V_k, diagonals, V_k_T_y_train
    )

    end = time.monotonic()
    q_vector_time = end - start
    if debug:
        logging.info(f"Q-Vector time: {q_vector_time:.3f}s")

    populate_q_vectors_dictionary(
        produce_voc_curve,
        block_sizes,
        voc_grid,
        is_multiclass,
        block,
        voc_curve,
        q_vector,
    )
    end_full_block = time.monotonic()
    full_block_time = end_full_block - start_full_block

    logging.info(f"Full block time: {full_block_time:.3f}s")
    return q_vector


def process_one_block_of_features(
        block: int,
        block_sizes: list,
        sample_size: int,
        voc_curve: dict,
        Vs: dict,
        eigens: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shrinkage_list: list,
        niu: int,
        seed: int,
        voc_grid: list,
        gpu: Optional[bool] = False,
        produce_voc_curve: Optional[bool] = False,
        debug: Optional[bool] = False,
        msrr: bool = False,
        factor_msrr: bool = False,
        date_indices_for_panel_dict: dict = {},
        pre_specified_list_of_specs_for_random_features: list = None
):
    """
    This is the main mathematical function that actually processes one block and updates the q-vector

    Parameters
    ----------
    block
    block_sizes
    sample_size
    voc_curve
    Vs
    eigens
    X_train
    y_train
    shrinkage_list
    niu
    seed
    voc_grid
    gpu
    produce_voc_curve
    debug
    msrr
    factor_msrr
    date_indices_for_panel_dict
    pre_specified_list_of_specs_for_random_features

    Returns
    -------

    """
    # Chunk Size
    p_1 = block_sizes[block + 1] - block_sizes[block]

    # 1. Generate Random Features
    # Should be T x P_1
    random_features = RandomFeaturesGeneratorr.generate_random_features_from_list_with_potential_ranking(
        seed=int((seed + 1) * 1e3) + block + 1,
        msrr=msrr,
        factor_msrr=factor_msrr,
        number_features_in_subset=p_1,
        pre_specified_list_of_specs_for_random_features=pre_specified_list_of_specs_for_random_features,
        date_ids=date_indices_for_panel_dict['train'],
        y_train=y_train,
        X_train=X_train
    )

    # Divide them by T^1/2
    random_features = random_features / np.sqrt(sample_size)
    is_multiclass = len(y_train.shape) > 1 and y_train.shape[1] > 1
    y_train_to_use = np.ones([random_features.shape[0], 1]) if msrr else y_train
    if block == 0:
        q_vector = produce_the_first_q_vector_for_zero_block(gpu,
                                                             random_features,
                                                             niu,
                                                             p_1,
                                                             debug,
                                                             is_multiclass,
                                                             y_train_to_use,
                                                             shrinkage_list,
                                                             produce_voc_curve,
                                                             block_sizes,
                                                             voc_grid,
                                                             block,
                                                             voc_curve,
                                                             Vs,
                                                             eigens)

    # block > 0
    else:
        q_vector = produce_q_vector_for_nonzero_block(gpu,
                                                      sample_size,
                                                      random_features,
                                                      niu,
                                                      p_1,
                                                      debug,
                                                      is_multiclass,
                                                      y_train_to_use,
                                                      shrinkage_list,
                                                      produce_voc_curve,
                                                      block_sizes,
                                                      voc_grid,
                                                      block,
                                                      voc_curve,
                                                      Vs,
                                                      eigens)
    return q_vector


def random_features_regression_giant_spectral_method(
        X_train: np.ndarray,
        y_train: np.ndarray,
        shrinkage_list: list,
        number_random_features: int,  # P
        small_subset_size: int,  # P'
        niu: int,
        seed: int,
        X_test: np.ndarray,
        y_test: np.ndarray,
        voc_grid: list,
        gpu: Optional[bool] = False,
        produce_voc_curve: Optional[bool] = False,
        produce_betas: Optional[bool] = False,
        debug: Optional[bool] = False,
        normalize_p: Optional[bool] = False,
        msrr: bool = False,
        factor_msrr: bool = False,
        date_indices_for_panel_dict: dict = {},
        parameters_for_random_features: tuple = None
) -> dict:
    """
        Important: the code assumes that stock ids are already sorted!
        Pre-process the data so that stock ids are increasing !!
        so the original data must first be sorted on dates.
        Then, conditional on any date we sort stock ids.
        And this pre-sorted data comes into the function

        Same for date_ids: We are appending data!
        So we assume that

        Parameters
        ----------
        y_test
        c_sigma
        list_of_width
        debug:
        date_indices_for_panel_dict: dictionary with 'test' and 'train', showing dates for train and test slices.
        Is important for factor construction. And is also important if we need to do cross-sectional ranking on each date
        factor_msrr: if True, then we build factors \sum_i R_i S_i(k), and then do MSRR on them
        msrr: If True, then we do msrr on R*S
        gpu: If True, then we use GPU for matrix multiplications
        normalize_p: if true, then we normalize features by sqrt(their number). Should be set to false
        gamma: list of gamma parameters used for random feature generation
        activation: activation function used for random feature construction
        voc_grid: grid for producing VOC curve. Must be multiples of small_subset_size
        produce_betas : If True, then we also output the giant beta vector.
        It could be huge (size = number_random_features, which could be a million or so)
        produce_voc_curve : If True, then we actually output predictions for a giant grid of numbers of random features
        (with a step size of roughly number_random_features / small_subset_size)
        niu: this is the main new parameter: the number of top \nu eigenvalues used
        X_test : the chunk of out-of-sample (test) data on which we produce OOS predictions
        shrinkage_list : list of ridge shrinkage basic_parameters
        X_train : in-sample raw signals from which random features are constructed
        y_train : in-sample returns to be predicted
        number_random_features : how many random features we want to produce. Could be a very large number
        small_subset_size : we split random features into sub-groups so that they fit in memory and
        running it becomes feasible even on a small machine
        seed : random seed. One should run this for a fixed seed, and then average predictions across seeds

        Returns dictionary
        output = {'rmt_stuff': rmt_stuff,
                  'betas': betas,
                  'future_predictions': future_predictions}

        -------
        rmt_stuff:  random matrix theory stuff that can be used to compute the
                    optimal shrinkage parameter z_*
        'betas':    actual regression betas (for each shrinkage level)
        'future_predictions':   Actual predictions for each (date-stock)
        Each of these is itself a dictionary, indexed by a grid of "numbers of random features"
        If produce_voc_curve = True, then this is an actual grid.
        If produce_voc_curve = False, then this is just one point, = number_random_features
        For each number_of_features, the corresponding future_predictions[number_of_features]
        is a matrix, dimension (OOS sample size) \times (shrinkage_list), so that for each value of
        the shrinkage parameter we have one prediction.

        Similarly for the rmt_stuff and betas.

        Why would the OOS sample size be big? Well, we do not need to
        re-compute betas every period.
        It is enough to do it every few month (say, every 3 months),
        in which case OOS sample size = number_of_stocks \times 3

        """
    t_first = time.time()
    if gpu:
        from utils.cp_linear_algebra import cp_eigh

    random_features_start = time.monotonic()
    sample_size = X_train.shape[0]

    block_sizes = get_block_sizes(
        number_random_features, small_subset_size, voc_grid
    )
    logging.info(f"# Block: {len(block_sizes)}")

    Vs = {}
    eigens = {}
    voc_curve = {}
    voc_curve["q_vectors"] = {}
    is_multiclass = len(y_train.shape) > 1 and y_train.shape[1] > 1

    for block in range(len(block_sizes) - 1):
        q_vector = process_one_block_of_features(
            block,
            block_sizes,
            sample_size,
            voc_curve,
            Vs,
            eigens,
            X_train,
            y_train,
            shrinkage_list,
            niu,
            seed,
            voc_grid,
            gpu,
            produce_voc_curve,
            debug,
            msrr=msrr,
            factor_msrr=factor_msrr,
            date_indices_for_panel_dict=date_indices_for_panel_dict,
            pre_specified_list_of_specs_for_random_features=parameters_for_random_features)

    if is_multiclass:
        voc_curve["q_vectors"].update({block_sizes[block + 1]: deepcopy(q_vector)})
    else:
        voc_curve["q_vectors"].update(
            {block_sizes[block + 1]: [deepcopy(q_vector)]}
        )

    # ****************
    # predictions
    # ****************

    (
        betas,
        predictions,
        future_random_features_all,
    ) = compute_betas_and_predictions(
        seed=seed,
        shrinkage_list=shrinkage_list,
        test_and_train={'test': X_test, 'train': X_train},
        block_sizes=block_sizes,
        y_train=y_train,
        y_test=y_test,
        voc_curve=voc_curve,
        produce_betas=produce_betas,
        test=False,
        normalize_p=normalize_p,
        msrr=msrr,
        factor_msrr=factor_msrr,
        date_indices_for_panel_dict=date_indices_for_panel_dict
    )
    strategy_returns = {key: transform_predictions_into_strategy_returns(predictions=predictions['test'][key],
                                                                         asset_returns=y_test,
                                                                         date_ids=date_indices_for_panel_dict['test'],
                                                                         shrinkage_list=shrinkage_list,
                                                                         msrr=msrr,
                                                                         factor_msrr=factor_msrr)
                        for key in predictions['test']}

    random_features_end = time.monotonic()
    t_last = time.time()
    print(f'-----------------------------total spectral function took {t_last - t_first}---------------------------')
    return {
        "betas": betas,
        "strategy_returns": strategy_returns,
        "future_random_features_all": future_random_features_all,
        "random_features_time": random_features_end - random_features_start,
    }


def run_giant_spectral_or_not_depending_on_size(slice_: dict,
                                                niu: int,
                                                giant_with_spectral_threshold: int = 3500,
                                                factor_msrr: bool = False):
    print(f'running next slice with {slice_["X_train"].shape[0]} sample size')

    if slice_['X_train'].shape[0] < giant_with_spectral_threshold or factor_msrr:
        # if factor_msrr, then the actual data we regress is small (T \times number of features;
        # unless we go to HFT, T is small and hence no need to use random_features_regression_giant_spectral_method)
        # if train sample is small, we can do simple stuff
        return ridge_regression_with_giant_number_of_random_features(**slice_)
    else:
        # otherwise we are forced to do complex spectral stuff
        slice_.update({'niu': niu})
        return random_features_regression_giant_spectral_method(**slice_)


def rolling_giant_panel_ridge_universal(
        signals: np.ndarray,
        returns: pd.DataFrame,
        date_ids: np.ndarray,
        window: int,
        diagonal_shrinkage: list,
        number_random_features: int,  # P
        small_subset_size: int,  # P'
        niu: int,
        seed: int,
        voc_grid: list,
        produce_voc_curve: Optional[bool] = False,
        produce_betas: Optional[bool] = False,
        lag: int = 1,
        stepp=30,
        normalize_before_ridge=True,
        maximal_fraction_of_nans_among_signals_for_a_given_ticker=0.8,
        giant_with_spectral_threshold: int = 4000,
        msrr: bool = False,
        factor_msrr: bool = False,
        parameters_for_random_features: list = None,
        ranking: str = None
) -> dict:
    """
    This function gets prediction of rolling ridge regression or a rolling msrr (which is still a ridge regression,
    but with 1 regressed on R*S
    :param msrr: In this case, we regress 1 on (R * S)
    :param stepp: how often we re-run the regression
    :param signals: signals, must be a dictionary indexed by tickers
    :param returns: returns, is a dataframe
    :param window: rolling window for the regression
    :param diagonal_shrinkage: lost of z parameters
    :param lag: lag =1 means there is a zero lag between running regression and trading. This is a
        partially unrealistic assumption in real world markets
    :return:
    """
    results = list()

    if number_random_features not in voc_grid:
        voc_grid = voc_grid + [number_random_features]

    input_for_the_ridge = {'number_random_features': number_random_features,
                           'small_subset_size': small_subset_size,
                           'seed': seed,
                           'voc_grid': voc_grid,
                           'produce_voc_curve': produce_voc_curve,
                           'produce_betas': produce_betas,
                           'msrr': msrr,
                           'factor_msrr': factor_msrr,
                           'parameters_for_random_features': (ranking, parameters_for_random_features),
                           'shrinkage_list': diagonal_shrinkage
                           }
    bad_dates = list()
    # first we slice the data into time slices

    """
    due to the size of the data and potentially very long rolling windows, 
    pre-creating all slices is too memory expensive  
    """
    unique_dates = np.unique(date_ids)
    for i in range(window + lag - 1, len(unique_dates), stepp):
        t1 = time.time()
        date = unique_dates[i]
        slice_ = get_a_single_panel_slice_with_ins_and_oos(
            signals=signals,
            returns_frame=returns,
            date_ids=date_ids,
            date=date,
            window=window,
            lag=lag,
            stepp=stepp,
            maximal_fraction_of_nans_among_signals_for_a_given_ticker
            =maximal_fraction_of_nans_among_signals_for_a_given_ticker,
            normalize_before_ridge=normalize_before_ridge,
            unique_dates=unique_dates
        )
        if slice_['X_train'].shape[0] < 100:
            bad_dates += [i]

        slice_.update(input_for_the_ridge)
        t2 = time.time()
        print(f'slicing took {t2 - t1}')
        results += [run_giant_spectral_or_not_depending_on_size(
            slice_=slice_,
            niu=niu,
            giant_with_spectral_threshold=giant_with_spectral_threshold,
            factor_msrr=factor_msrr)]

    # now we take all the slices and concatenate them into a pd.DataFrame of predictions
    # each column corresponds to a different way of predicting
    strategy_returns = {key: pd.concat([result['strategy_returns'][key]
                                        for result in results], axis=0).sort_index() for key in
                        results[0]['strategy_returns']}
    return strategy_returns


def update_the_psi_matrix(gpu: bool,
                          psi_matrix: np.ndarray,
                          random_features: np.ndarray):
    if psi_matrix is None:
        psi_matrix = np.zeros([random_features.shape[0], random_features.shape[0]])
    if gpu:  # make it parametric:
        torch.cuda.empty_cache()
        psi_matrix += (
            torch.matmul(
                torch.tensor(random_features).to(device),
                torch.tensor(random_features.T).to(device),
            )
            .cpu()
            .numpy()
        )
    else:
        psi_matrix += random_features @ random_features.T
    return psi_matrix


def giant_msrr(seed: int,
               number_features_per_ticker: int,
               slice_: dict,
               shrinkage_list: list,
               test_mode: bool = True,
               gpu: bool = False,
               normalize_p: bool = False,
               pre_specified_list_of_specs_for_random_features: list = None,
               seed_step: int = 1e3,
               produce_betas: bool = False,
               use_original_signals: bool = False
               ):
    random_features_all = []
    returns_in_sample = slice_['train_ret'].values
    sigs_slices_in_sample = slice_['train_sigs']

    print_header("Computing Psi Matrix and Q vector")
    psi_matrix = np.zeros([returns_in_sample.shape[0], returns_in_sample.shape[0]])
    for ii, ticker in enumerate(slice_['train_ret'].columns):
        random_features = (
            RandomFeaturesGeneratorr.generate_random_features_from_lisst(
                seed=int((seed + 1) * seed_step) + ii,
                # important: here, there is no need to re-generate a different seed for each ticker
                features=sigs_slices_in_sample[ticker].values,
                pre_specified_list_of_specs=pre_specified_list_of_specs_for_random_features,
                number_features_in_subset=number_features_per_ticker,
                just_return_original_features=use_original_signals
            )
        )
        # now we build managed returns
        random_features = random_features * returns_in_sample[:, ii].reshape(-1, 1)

        if test_mode:
            random_features_all.append(random_features)

        # this is the main bottleneck for big matrix
        psi_matrix = update_the_psi_matrix(gpu,
                                           psi_matrix,
                                           random_features)
    q_vector, psi_hat_eig = build_the_q_vector(
        psi_matrix,
        np.ones([returns_in_sample.shape[0], 1]),  # ones because we are regressing 1 on managed returns
        shrinkage_list,
        number_features_per_ticker,
        normalize_p=normalize_p
    )

    if test_mode:
        random_features = np.concatenate(random_features_all, axis=1)
        # Covariance matrix
        true_psi_matr = random_features.T @ random_features / random_features.shape[0]

        betas_direct = simple_markowitz(true_psi_matr,
                                        random_features,
                                        shrinkage_list)
    else:
        true_psi_matr = None
        random_features = None

    strategy_returns, betas = compute_strategy_returns_for_giant_msrr(
        tickers=slice_['train_ret'].columns,
        number_features_per_ticker=number_features_per_ticker,
        pre_specified_list_of_specs_for_random_features=pre_specified_list_of_specs_for_random_features,
        q_vector=q_vector,
        slice_=slice_,
        seed=seed,
        seed_step=seed_step,
        shrinkage_list=shrinkage_list,
        use_original_signals=use_original_signals
    )
    breakpoint()
    return strategy_returns


def get_slices_for_tickers_with_enough_data(dates,
                                            i: int,
                                            window: int,
                                            lag: int,
                                            returns: pd.DataFrame,
                                            signals: dict,
                                            maximal_fraction_of_nans: float,
                                            normalize_before_ridge: bool,
                                            stepp: int,
                                            make_data_balanced: bool = False):
    slice_ = dict()
    first_date = dates[i - window - lag + 1]
    last_date = dates[i - lag + 1]
    rets = returns.loc[first_date:last_date]
    tickers_with_enough_returns = rets.isna().mean() < maximal_fraction_of_nans
    if make_data_balanced:
        tmp = {ticker: signals[ticker].loc[first_date:last_date, :].isna().mean().mean() for ticker in signals}
        tmp = pd.DataFrame.from_dict(tmp, orient='index').reindex(tickers_with_enough_returns.index)
        tmp = (tmp < maximal_fraction_of_nans)
        tickers_with_enough_data = tickers_with_enough_returns * tmp.values.flatten()
    else:
        tickers_with_enough_data = tickers_with_enough_returns

    rets = rets.loc[:, tickers_with_enough_data].fillna(0)
    sigs = {ticker: signals[ticker].loc[first_date:last_date, :] for ticker in rets.columns}

    if make_data_balanced:
        nans = {ticker: sigs[ticker].loc[first_date:last_date, :].isna().mean(0) for ticker in rets.columns}
        signals_with_enough_data = pd.DataFrame.from_dict(nans, orient='index').mean(0) < maximal_fraction_of_nans

        sigs = {ticker: sigs[ticker].loc[first_date:last_date, signals_with_enough_data].fillna(0)
                for ticker in rets.columns}

    else:
        sigs = {ticker: sigs[ticker].loc[first_date:last_date,
                        sigs[ticker].isna().mean() < maximal_fraction_of_nans].fillna(0)
                for ticker in rets.columns}

    if normalize_before_ridge:
        sigs = {ticker: (sigs[ticker] / np.sqrt((sigs[ticker] ** 2).mean())).fillna(0) for ticker in sigs}

    slice_['train_ret'] = rets
    slice_['train_sigs'] = sigs

    first_date = dates[i + 1]
    last_date = dates[min(i + 1 + stepp, len(dates) - 1)]
    oos_rets = returns.loc[first_date:last_date, rets.columns].fillna(0)
    # it is important that we use the same signals as in - sample
    oos_sigs = {ticker:
                    signals[ticker].loc[first_date:last_date,
                    sigs[ticker].columns].fillna(0).replace([np.inf, -np.inf], 0) for ticker in
                rets.columns}
    if normalize_before_ridge:
        # it is important that we normalize by the in-sample variance
        oos_sigs = {ticker: (oos_sigs[ticker]
                             / np.sqrt((sigs[ticker] ** 2).mean())).fillna(0).replace([np.inf, -np.inf], 0) for ticker
                    in oos_sigs}

    slice_['test_ret'] = oos_rets
    slice_['test_sigs'] = oos_sigs
    return slice_


def rolling_giant_msrr_universal(
        signals: dict,
        returns: pd.DataFrame,
        window: int,
        diagonal_shrinkage: list,
        number_random_features_per_ticker: int,  # P
        seed: int,
        produce_betas: Optional[bool] = False,
        lag: int = 1,
        stepp=30,
        normalize_before_ridge=True,
        maximal_fraction_of_nans=0.3,
        parameters_for_random_features: list = None,
        t_cost_optimization: bool = False,
        use_original_signals: bool = False,
        make_balanced_data: bool = False
) -> tuple:
    """
    This function gets prediction of rolling ridge regression
    :param stepp:
    :param signals:
    :param returns:
    :param window:
    :param diagonal_shrinkage:
    :param lag: lag =1 means there is a zero lag between runnung regression and trading. This is a
        partially unrealistic assumption in real world markets
    :return:
    """
    dates = returns.index
    # first we slice the data into time slices

    """
    due to the size of the data and potentially very long rolling windows, 
    pre-creating all slices is too memory expensive  
    """

    input_for_the_ridge = {'number_features_per_ticker': number_random_features_per_ticker,
                           'seed': seed,
                           'produce_betas': produce_betas,
                           'shrinkage_list': diagonal_shrinkage,
                           'pre_specified_list_of_specs_for_random_features': parameters_for_random_features
                           }
    if t_cost_optimization:
        input_for_the_ridge.pop('produce_betas')

    results = list()

    for i in range(window + lag - 1, returns.shape[0], stepp):
        t1 = time.time()
        slice_ = get_slices_for_tickers_with_enough_data(dates,
                                                         i,
                                                         window,
                                                         lag,
                                                         returns,
                                                         signals,
                                                         maximal_fraction_of_nans,
                                                         normalize_before_ridge,
                                                         stepp,
                                                         make_data_balanced=make_balanced_data)
        # make_data_balanced=t_cost_optimization is because I would like to be able to do both factors and msrr
        if slice_['train_ret'].shape[1] < 2:
            """
            It means we have less than two tickers to trade with enough train data
            """
            continue

        input_for_the_ridge['slice_'] = slice_
        input_for_the_ridge['use_original_signals'] = use_original_signals

        if t_cost_optimization:
            input_for_the_ridge['tcosts'] = {ticker: 0.0001 for ticker in returns.columns}
            input_for_the_ridge['scales_list'] = [0, 0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]

            smooth_positions_naively(returns.fillna(0) / 1000,
                                     transaction_costs=input_for_the_ridge['tcosts'],
                                     positions=returns.fillna(0),
                                     scales=[10000000],
                                     shrinkage_list=[diagonal_shrinkage[0]],
                                     rolling_window=200,
                                     starting_point=500)

        t2 = time.time()
        print(f'preparing the slice took {t2 - t1}')
        print(f'running next slice with')
        if t_cost_optimization:
            results += [giant_msrr_with_t_cost(**input_for_the_ridge)]
        else:
            results += [giant_msrr(**input_for_the_ridge)]
    if t_cost_optimization:
        strategy_returns = pd.concat([result['strategy_returns'] for result in results], axis=0)
    else:
        strategy_returns = pd.concat(results, axis=0)
    # sharpe_ratio(strategy_returns, 1)

    # 'beta_names' will be absent when compute_smart_weights = False
    cols = results[0]['beta_names'] if 'beta_names' in results[0].keys() else diagonal_shrinkage

    # now we take all the slices and concatenate them into a pd.DataFrame of predictions
    # each column corresponds to a different way of predicting
    return strategy_returns


def get_slice_of_a_panel(returns: np.ndarray,
                         signals: np.ndarray,
                         date_ids: np.ndarray,
                         first_date: pd.Timestamp,
                         last_date: pd.Timestamp,
                         maximal_fraction_of_nans_among_signals_for_a_given_ticker: float,
                         ) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    this is the key pre-filtering algorithm
    We extract the slice for in-sample and for out-of-sample using the same!! criterion based on
    maximal_fraction_of_nans_among_signals_for_a_given_ticker.
    IT IS ABSOLUTELY CRUCIAL THAT WE USE THE SAME CRITERION, otherwise we cannot use train-sample to
    predict test sample!
    Parameters
    ----------
    returns
    signals
    date_ids
    first_date
    last_date
    maximal_fraction_of_nans_among_signals_for_a_given_ticker

    Returns
    -------

    """

    rets = returns[(date_ids >= first_date) & (date_ids < last_date)]
    sigs = signals[(date_ids >= first_date) & (date_ids < last_date), :]

    # ******************************
    # we only optimize over stuff for which we have enough data
    # ******************************
    good_indices = (np.isnan(sigs).sum(1) < maximal_fraction_of_nans_among_signals_for_a_given_ticker * sigs.shape[
        1]) * (np.std(sigs, axis=1) > 0)
    date_ids_selected = np.array(date_ids[(date_ids >= first_date) & (date_ids < last_date)])[good_indices]

    used_signals = sigs[good_indices, :]
    used_signals[np.isnan(used_signals)] = 0

    used_rets = rets[good_indices]
    used_rets[np.isnan(used_rets)] = 0
    # now we ceate the future data that will be used for the strategy creation
    # now we have a giant vector of (NT)\times 1 rets
    # and a giant array of (NT) \times P of signals
    return used_rets, used_signals, date_ids_selected


def get_a_single_panel_slice_with_ins_and_oos(signals: np.ndarray,
                                              returns_frame: pd.DataFrame,
                                              date_ids: np.ndarray,
                                              unique_dates: np.ndarray,
                                              date,
                                              window: int,
                                              lag: int,
                                              stepp: int,
                                              maximal_fraction_of_nans_among_signals_for_a_given_ticker: float = 0.85,
                                              normalize_before_ridge: bool = False
                                              ):
    '''
        This function get data slices including returns and signals.
        These signals must be a dictionary indexed by the ticker; ticker is columns of returns
        IT IS CRUCIAL THAT RETURNS ARE IN THE RIGHT ORDER ALIGNED WITH TICKERS !!
        :param signals:
        :param returns_frame:
        :param date_ids: list of dates that corresponds to the dates in the signals and returns data frame
                        (they should be the same indexed)
        :param date: current date
        :param window: rolling window size
        :param lag: lag =1 means there is a zero lag between runnung regression and trading. This is a
        partially unrealistic assumption in real world markets
        :param stepp:
        :param shrinkage_list:
        :param use_msrr:
        :param maximal_fraction_of_nans_among_signals_for_a_given_ticker:
        :param normalize_before_ridge:
        :param compute_smart_weights:
        :param predict_once: If predict_once is True, then future_signals only contains one observation
        :return:
        '''
    if lag < 1:
        print('hey, lag must be positive')
        return
    returns = returns_frame.values

    i = list(unique_dates).index(date)

    time_periods = len(unique_dates)
    date_indices_for_panel_dict = dict()
    rets, sigs, date_ids_ins = get_slice_of_a_panel(
        returns,
        signals,
        date_ids=date_ids,
        first_date=unique_dates[i - window - lag + 1],
        last_date=unique_dates[i - lag + 1],
        maximal_fraction_of_nans_among_signals_for_a_given_ticker
        =maximal_fraction_of_nans_among_signals_for_a_given_ticker
    )

    future_returns, future_signals, date_ids_oos = get_slice_of_a_panel(
        returns,
        signals,
        date_ids=date_ids,
        first_date=unique_dates[i],
        last_date=unique_dates[min(i + stepp, time_periods - 1)],
        maximal_fraction_of_nans_among_signals_for_a_given_ticker
        =maximal_fraction_of_nans_among_signals_for_a_given_ticker
    )

    date_indices_for_panel_dict['train'] = date_ids_ins
    date_indices_for_panel_dict['test'] = date_ids_oos
    if normalize_before_ridge:
        used_signals_std = np.sqrt(np.nanmean(sigs ** 2, axis=0))
        used_signals_std[used_signals_std < 10 ** (-10)] = 1
        sigs /= used_signals_std
        future_signals /= used_signals_std

    output = {'X_train': sigs,
              'y_train': rets,
              'X_test': future_signals,
              'y_test': future_returns,
              'date_indices_for_panel_dict': date_indices_for_panel_dict
              }

    return output


def create_panel_data_from_raw_signals(stevens_futures: Futures,
                                       signals_dict: dict,
                                       small_data_for_quick_test: int = None,
                                       rank_raw_features: str = None):
    """

    Parameters
    ----------
    stevens_futures: futures dataset
    signals_dict: dictionary of signals, indexed by ticker
    small_data_for_quick_test: to cut small data chunks for quick experiments
    rank_raw_features: an input to the function smart_rank(); defines how we rank the data

    Returns
    -------

    """
    number_tickers = len(list(signals_dict.keys()))
    for ticker in signals_dict:
        signals_dict[ticker]['returns'] = stevens_futures.returns[ticker]
        signals_dict[ticker]['ticker'] = ticker

    signals = pd.concat([signals_dict[ticker] for ticker in stevens_futures.returns.columns], axis=0)
    signals = signals.reset_index().sort_values('date').set_index(['date', 'ticker'])

    # for at least two different signals there exist at least 4 tickers to do cross-sectional trading
    minimal_number_of_tickers = 4
    minimal_number_of_signals_per_ticker = 2
    enough = (signals.isna().groupby(signals.index.get_level_values('date')).transform('sum')
              <= number_tickers - minimal_number_of_tickers).sum(1) > minimal_number_of_signals_per_ticker

    signals = signals[enough]
    returns = signals.pop('returns')
    if small_data_for_quick_test:
        returns = returns.iloc[:small_data_for_quick_test]
        signals = signals.iloc[:small_data_for_quick_test, :]

    date_ids = returns.index.get_level_values('date').values
    if rank_raw_features is not None:
        ranked_signals = rank_features_cross_sectionally(signals.values,
                                                         date_ids,
                                                         rank_raw_features)
    else:
        ranked_signals = signals.values
    return returns, ranked_signals, date_ids


def giant_trading(regression_type: str,
                  rolling_window: int,
                  voc_grid: list,
                  stepp: int,
                  giant_with_spectral_threshold: int = 3500,
                  number_random_features: int = 10000,
                  small_subset_size: int = 2000,
                  niu: int = 2000,
                  parameters_for_random_features: list = None,
                  small_data_for_quick_test: int = None,
                  rank_raw_features: str = None,
                  ranking_random_features: str = False,
                  maximal_fraction_of_nans_among_signals_for_a_given_ticker: float = 0.3,
                  t_cost_optimization: bool = False,
                  use_original_signals: bool = False,
                  make_balanced_data: bool = False
                  ):
    """
    This is the main routine for running different version of MSRR with a very large number of random features
    :param regression_type: We have
    'giant_panel_regression' : just a panel regression on random features

    'giant_panel_msrr': We create a panel of R_i * S
    And then we run a panel regression of 1 on R*S

    'giant_panel_factor_msrr': We first create factors F = \sum_i R_i S_i(k) where k is k-th random feature;
    and then we regress 1 on F. If there are millions of F, then this is still computationally very demanding

    'giant_msrr': is special. Here, there are no blocks of random features. Instead, say, for 100 or 1000 of securities,
    for each security we build its own set of, say, 10000 RFs. This makes it again memory infeasible.
    To deal with this issue, we create block that are per ticker: each block is a ticker.
    This requires a special function rolling_giant_msrr_universal which has a different logic

    :param rolling_window: rolling window for the regression
    :param voc_grid: grid of numbers of random features over which we run the strategy.
    If we want to have the strategy evaluated for several levels of complexity, then voc_grid drastically
    simplifies and speeds up this analysis: Due to the algorithm, we compute random features in chunks.
    So, we have to compute the voc_grid (voc=virtue of complexity) in any case

    Thus, if we want to have the strategy run for random features = [10k, 20k, 100k], no need for you to
    run the code three times. Run it once, with voc_grid=[10k, 20k, 100k]

    :param stepp: how often we re-run the regression
    :param giant_with_spectral_threshold: we have two methods for computing the regression.
    When the sample size is large, we use spectral decompositions of a special kind
    giant_with_spectral_threshold defines the sample size beyond which we use the complex spectral algorithm

    :param number_random_features: the total number of random features we will produce per ticker
    :param small_subset_size: think is the chunk size; we produce random features in chunks until we get to number_random_features
    :param niu: number of eigenvalues we use for the regression in the complex method
    :param parameters_for_random_features: parameters of random features generator (e.g., activation function,
    distribution, etc)
    :param ranking_random_features : can take two values (or None): 'rank' or 'cdf'. These are inputs to the smart_rank function
    :param rank_raw_features : can take two values (or None): 'rank' or 'cdf'. These are inputs to the smart_rank function
    :param small_data_for_quick_test: cut the initial small piece of data for quick tests
    :return:

    """
    stevens_futures = Futures(horizon=1,
                              merge_CME_SP_and_CME_ES_by_dollar_volume=True)
    signal_list = list(stevens_futures.all_signals.keys())
    signals_dict = {
        ticker: stevens_futures.get_all_signals_for_a_ticker(ticker)[signal_list]
        for ticker in stevens_futures.returns.columns}

    if regression_type != 'giant_msrr':
        returns, ranked_signals, date_ids = create_panel_data_from_raw_signals(
            stevens_futures,
            signals_dict,
            small_data_for_quick_test,
            rank_raw_features)

    # now signals_dict is a panel, just like stocks !
    if regression_type.startswith('giant_panel'):
        factor_msrr = False
        if regression_type == 'giant_panel_regression':
            msrr = False
        elif regression_type == 'giant_panel_msrr':
            msrr = True
        elif regression_type == 'giant_panel_factor_msrr':
            factor_msrr = True
            msrr = True
        rolling_giant_panel_ridge_universal(
            signals=ranked_signals,
            returns=returns,
            date_ids=date_ids,
            window=rolling_window,
            diagonal_shrinkage=stevens_futures.settings.shrinkage_list,
            lag=2,  # lag=1 is unrealistic, assumption zero gap between regression running and trading
            stepp=stepp,
            normalize_before_ridge=True,
            maximal_fraction_of_nans_among_signals_for_a_given_ticker
            =maximal_fraction_of_nans_among_signals_for_a_given_ticker,
            giant_with_spectral_threshold=giant_with_spectral_threshold,
            number_random_features=number_random_features,
            small_subset_size=small_subset_size,
            niu=niu,
            produce_voc_curve=True,
            seed=0,
            voc_grid=voc_grid,
            msrr=msrr,
            factor_msrr=factor_msrr,
            parameters_for_random_features=parameters_for_random_features,
            ranking=ranking_random_features
        )

    elif regression_type == 'giant_msrr':

        rolling_giant_msrr_universal(
            signals=signals_dict,
            returns=stevens_futures.returns,
            window=rolling_window,
            diagonal_shrinkage=stevens_futures.settings.shrinkage_list,
            lag=2,  # lag=1 is unrealistic, assumption zero gap between regression running and trading
            stepp=stepp,
            normalize_before_ridge=True,
            maximal_fraction_of_nans=maximal_fraction_of_nans_among_signals_for_a_given_ticker,
            seed=0,
            parameters_for_random_features=parameters_for_random_features,
            number_random_features_per_ticker=number_random_features,
            t_cost_optimization=t_cost_optimization,
            use_original_signals=use_original_signals,
            make_balanced_data=make_balanced_data
        )


if __name__ == '__main__':
    # export PYTHONPATH="${PYTHONPATH}:/Users/malamud/Dropbox/MY_STUFF/TRADING/InfiniteRandomFeaturesForTeaching"
    small_subset_size = 200

    # here it is important that we generate random features in small chunks, as many as the small_subset_size
    types, parameters = generate_parameters(
        number_simulations=1,  # this is the number of random seeds
        gamma_values=[1, 2, 3],
        number_features_per_gamma=small_subset_size)

    regtype = 'giant_msrr'
    # 'giant_panel_regression'  # 'giant_msrr'  # 'giant_panel_regression'
    # 'giant_msrr'  # 'giant_panel_regression'  # 'giant_panel_factor_msrr'

    giant_trading(regression_type=regtype,
                  rolling_window=100,
                  voc_grid=[6000, 8000, 10000],
                  stepp=50,
                  small_subset_size=small_subset_size,
                  parameters_for_random_features=parameters,
                  small_data_for_quick_test=20000,
                  rank_raw_features='rank',
                  ranking_random_features='rank',
                  t_cost_optimization=True,
                  use_original_signals=True,
                  make_balanced_data=True
                  )
