import numpy as np

from helpers.auxilliary_functions import simple_markowitz


def transform_sigma(sigma: np.ndarray):
    """
    we compute 0.5 (hatSigma - (hatSigma^2-4I)^{1/2})
    :param sigma:
    :return:
    """
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    tmp = 0.5 * eigenvectors @ ((eigenvalues - np.sqrt(eigenvalues ** 2 - 4)).reshape(-1, 1) * eigenvectors.T)
    return tmp


def compute_tildem(xi: np.ndarray,
                   sigma_hat: np.ndarray):
    """
    We compute tilde m for the case when G = xixi'.
    Then,
    tilde m = (xi)^{-1/2} 0.5 (hatSigma - (hatSigma^2-4I)^{1/2}) (xi)^{-1/2}
    :param xi:
    :param sigma_hat:
    :return:
    """
    tmp = transform_sigma(sigma_hat)

    tilde_m = normalize_sigma_by_lambda(xi,
                                        tmp)
    return tilde_m


def normalize_sigma_by_lambda(lam: np.ndarray,
                              sigma: np.ndarray):
    """
    we compute Lambda^{-1/2} Sigma Lambda^{-1/2}
    :param lam:
    :param sigma:
    :return:
    """
    norm = lam ** (- 1 / 2)
    tmp = norm.reshape(-1, 1) * (sigma * norm.reshape(1, -1))
    return tmp


def compute_sigma_hat(lam: np.ndarray,
                      sigma: np.ndarray,
                      xi: np.ndarray):
    """
    hatSigma = xi^{-1/2} (Lambda^{-1/2} Sigma Lambda^{-1/2} + diag(xi^2+1)) xi^{-1/2}
    :param lam:
    :param sigma:
    :param xi:
    :return:
    """
    normalized_sigma = normalize_sigma_by_lambda(lam,
                                                 sigma)

    tmp = normalized_sigma + np.diag(xi.flatten() ** 2 + 1)

    sigma_hat = normalize_sigma_by_lambda(xi,
                                          tmp)
    return sigma_hat


def get_m_from_tilde_m(lam: np.ndarray,
                       tilde_m: np.ndarray):
    """
    m = Lambda^{-1/2} m Lambda^{1/2}
    :param lam:
    :param tilde_m:
    :return:
    """
    m_ = (lam ** (- 1 / 2)).reshape(-1, 1) * tilde_m * (lam ** (1 / 2)).reshape(1, -1)
    return m_


def initial_point_m(lam: np.ndarray,
                    sigma: np.ndarray,
                    xi: np.ndarray):
    """
    This is how we compute the initial point for iterations that eventually converge to
    the true optimal m
    :param lam:
    :param sigma:
    :param xi:
    :return:
    """
    sigma_hat = compute_sigma_hat(lam,
                                  sigma,
                                  xi)
    tilde_m = compute_tildem(xi, sigma_hat)

    return tilde_m


def verify_equation_for_m_and_m_til(lam: np.ndarray,
                                    capital_g: np.ndarray,
                                    m_: np.ndarray,
                                    sigma: np.ndarray):
    bar_lam = np.diag(lam.flatten() * np.diag(capital_g))
    denominator = sigma + np.diag(lam.flatten()) + bar_lam
    diff = denominator @ m_ - (((lam.reshape(-1, 1) * m_) * capital_g) @ m_ + np.diag(lam.flatten()))
    print(f'the error is {np.mean(np.abs(diff))}')

    return diff


def map_f_for_tilde_m(lam: np.ndarray,
                      capital_g: np.ndarray,
                      tilde_m: np.ndarray,
                      sigma: np.ndarray,
                      shrinkage: float):
    """
    Here I am doing a bit of shrinkage and a bit of compute cost optimization

    :param lam:
    :param capital_g:
    :param tilde_m:
    :param sigma:
    :param g_matrix:
    :return:
    """
    # first we compute Lambda^{1/2} (I - tilde_m) Lambda^{1/2}
    tmp = normalize_sigma_by_lambda(1 / lam, np.eye(tilde_m.shape[0]) - tilde_m)
    tmp1 = (tmp * capital_g)
    tmp2 = normalize_sigma_by_lambda(lam, sigma + tmp1)

    # note that we are adding 1 here. In the paper, we do (I+tmp2)^{-1}. Here, we also
    # add a shrinkage z and do (I (1+z) +tmp2)^{-1}
    tmp = np.linalg.inv(np.eye(tmp2.shape[0]) * (1 + shrinkage) + tmp2)
    return tmp


def solve_m_tilde_through_iterations(g_bar: np.ndarray,
                                     capital_g: np.ndarray,
                                     sigma: np.ndarray,
                                     initial_value: str,
                                     num_iterations: int,
                                     lam: np.ndarray,
                                     scale_for_lam: float,
                                     shrinkage: float):
    """

    :param sigma: covariance matrix of returns
    :param capital_g:
    :param num_iterations:
    :param lam:
    :param scale_for_lam:
    :param shrinkage_list:
    :param g_bar:
    :param covariance_matrix:
    :param scale_for_g:
    :param initial_value:
    :return:
    """
    if initial_value == 'low' or g_bar.min() < 0.001:
        """
        with negative expected returns, numerical approximation for initial_point_m 
        does not work, so we still need to use the low one
        """
        m_til = 0 * capital_g
    else:
        # then we use for the initial point the smart one, with just bar g
        m_til = initial_point_m(scale_for_lam * lam,
                                sigma=sigma,
                                xi=g_bar)

    for iter in range(num_iterations):
        old_m_til = m_til.copy()
        m_til = map_f_for_tilde_m(lam=scale_for_lam * lam,
                                  capital_g=capital_g,
                                  tilde_m=old_m_til,
                                  sigma=sigma,
                                  shrinkage=shrinkage
                                  )

    m_ = get_m_from_tilde_m(lam=lam,
                            tilde_m=m_til)

    return m_til, m_


def get_m_matrix_for_slice(returns: np.ndarray,
                           lam: np.ndarray,
                           risk_aversion: float,
                           lam_scale: float,
                           start_low: bool = True,
                           shrinkage_for_m: float = 0.,
                           use_naive_m_matrix: bool = False
                           ):
    returns = 1 + returns

    # the clipping below is just in case, to avoid crazy outliers etc
    bar_g = returns.mean(0).clip(lower=0.5, upper=1.5).reshape(-1, 1)

    covariance_matrix = np.cov(returns)
    if use_naive_m_matrix:
        # naive m matrix is (Sigma + z + Lambda)^{-1} Lambda
        m_ = np.linalg.inv(covariance_matrix + shrinkage_for_m * np.eye(covariance_matrix.shape[0])
                           + lam_scale * np.diag(lam)) * lam_scale * lam.reshape(1, -1)
        return m_, covariance_matrix, bar_g

    second_moment_matrix = covariance_matrix + bar_g @ bar_g.T

    _, m_ = solve_m_tilde_through_iterations(g_bar=bar_g,
                                             capital_g=second_moment_matrix,
                                             sigma=covariance_matrix * risk_aversion,
                                             initial_value='low' if start_low else 'high',
                                             num_iterations=10,
                                             lam=lam,
                                             scale_for_lam=lam_scale,
                                             shrinkage=shrinkage_for_m)

    return m_, covariance_matrix, bar_g


def true_aim(markowitz_portfolio_matrix: np.ndarray,
             m_: np.ndarray,
             lam: np.ndarray,
             g_bar: np.ndarray,
             sigma: np.ndarray
             ):
    """
    :param markowitz_portfolio_matrix:
    For parametric portfolios (MSRR and alike),
    Markowitz is itself parametrix, given by B * S_t
    where B is some complicated matrix (that we call the "Markowitz Portfolio Matrix")
    Note that
    :param m_: the m matrix
    :param lam: transaction costs
    :param g_bar: mean portfolio growth
    :param sigma: covariance matrix of returns
    :return:
    """
    c_matrix = m_ @ ((1 / lam).reshape(-1, 1) * sigma)

    # mathematically, it is possible to show that the norm_matrix below is close to I when
    # g_matix is close to I
    # (this is the case for additive cost models where position drift is not important)

    norm_matrix = np.linalg.inv(np.eye(m_.shape[0]) - m_) \
                  @ np.linalg.inv(np.eye(m_.shape[0]) - m_ * g_bar.reshape(1, -1)) @ c_matrix

    return norm_matrix @ markowitz_portfolio_matrix


def new_portfolio_from_old(m_: np.ndarray,
                           old_portfolio: np.ndarray,
                           aim: np.ndarray,
                           g_: np.ndarray):
    new_portfolio_matrix = m_ @ (g_ * old_portfolio) + (np.eye(m_.shape[0]) - m_) @ aim
    return new_portfolio_matrix


def build_ml_tcost_aware_portfolio_for_slice(returns_slice: np.ndarray,
                                             signals_indexed_by_time: dict,
                                             list_of_shrinkages: list,
                                             dates: list,
                                             lam: np.ndarray,
                                             markowitz_portfolio_matrix: np.ndarray,
                                             use_msrr_logic_for_sigma: bool = True,
                                             initial_point_spec: str = 'use_zero',
                                             risk_aversion: float = 1.,
                                             lam_scale: float = 1.,
                                             use_naive_m_matrix: bool = False):
    """
    WARNING: SIGNALS ARE ASSUMED TO BE ALREADY LAGGED !
    :param lam: transaction costs
    :param dates: dates for the slice
    :param list_of_shrinkages: final shrinkage parameters
    :param risk_aversion: parameter by which we multiply covariance matrix
    :param lam_scale: scale of Lambda (another shrinkage parameter)
    :param use_naive_m_matrix: If True, then we build a trivial m matrix that is completely myopic
    :param use_msrr_logic_for_sigma: This is an important boolean parameter.
    Manged portfolio logic creates R_{t+1}S_t and views them as assets.
    This is kind of equivalent to using R_{t+1} R_{t+1}' instead of Sigma
    for the covariance matrix
    :param markowitz_portfolio_matrix:
    :param returns_slice: actual stock returns. To be used for estimating means and covariances
    :param signals_indexed_by_time: it is convenient to have signals ready as a dictionary indexed by time,
    to speed up stuff.
    :param initial_point_spec: can take three values for the moment:
    'use_zero' means we use zero
    'myopic_aim' means we use myopic Aim using the current target portfolio.
    'average_aim' means we use historical average of the target portfolio
    This is a subtle question of what specification here gives the right way to measure correct
    "long-run" transaction costs
    :return:
    """
    # todo for the moment, this logic is highly inefficient in high-dimensional settings
    # todo to be redone using the chunk logic
    T_ = returns_slice.shape[0]

    m_, covariance_matrix, g_bar = get_m_matrix_for_slice(returns_slice,
                                                          lam=lam,
                                                          risk_aversion=risk_aversion,
                                                          lam_scale=lam_scale,
                                                          start_low=True,
                                                          shrinkage_for_m=0.,
                                                          use_naive_m_matrix=use_naive_m_matrix
                                                          )
    # now we build the object that we call Pi_t in the paper. It is a matrix times signals[t_]
    initial_point = 0 * signals_indexed_by_time[dates[0]]

    if initial_point_spec == 'myopic_aim':
        initial_point = markowitz_portfolio_matrix @ signals_indexed_by_time[dates[0]]
    elif initial_point_spec == 'average_aim':
        average_signals = signals_indexed_by_time[dates[0]] * 0
        for t_, date in enumerate(dates):
            average_signals += signals_indexed_by_time[date]
        average_signals /= len(dates)
        initial_point = markowitz_portfolio_matrix @ average_signals

    num_signals = signals_indexed_by_time[0].shape[1]

    # we now use the portfolio approach: reduce the problem to find the weights of the
    # parametric portfolio

    average_r_pi = np.zeros([num_signals, 1])
    average_pi_sigma_pi = np.zeros([num_signals, num_signals])

    new_portfolio_matrix = initial_point
    for t_ in range(1, len(dates)):
        old_portfolio_matrix = new_portfolio_matrix.copy()
        # todo this can massively optimized with chunk logic
        # careful, signals are already assumed to be
        aim = true_aim(markowitz_portfolio_matrix,
                       m_,
                       lam * lam_scale,
                       g_bar,
                       covariance_matrix * risk_aversion) @ signals_indexed_by_time[t_]
        # so aim = complicated_matrix @ signals[t_]
        g_ = (1 + returns_slice[t_, :].clip(-0.5, 0.5).reshape(-1, 1))
        new_portfolio_matrix = new_portfolio_from_old(m_,
                                                      old_portfolio_matrix,
                                                      aim,
                                                      g_)

        average_r_pi += returns_slice[t_, :].reshape(1, -1) @ new_portfolio_matrix
        if use_msrr_logic_for_sigma:
            sig = risk_aversion * covariance_matrix
        else:
            sig = risk_aversion * returns_slice[t_, :].reshape(-1, 1) * returns_slice[t_, :].reshape(1, -1)

        tmp = new_portfolio_matrix.T @ sig @ new_portfolio_matrix

        # this is by how much the position is changing
        diff = new_portfolio_matrix - g_.reshape(-1, 1) * old_portfolio_matrix
        tmp += diff.T * (lam.reshape(-1, 1) * diff) * lam_scale
        # this is the parametric portfolio formula from the paper
        average_pi_sigma_pi += tmp
    # the last aim in this loop is the aim at the last available date

    last_aim = aim
    average_r_pi /= len(dates)
    average_pi_sigma_pi /= len(dates)
    betas = simple_markowitz(matrix=average_pi_sigma_pi,
                             returns=average_r_pi,
                             shrinkage_list=list_of_shrinkages,
                             ready_means=average_r_pi)
    # the last aim
    final_aim = last_aim @ betas  # this is basically our ansatz: aim = matrix * S_t * beta
    return m_, final_aim


def experiment_with_m():
    np.random.seed(0)
    returns = 1 + np.random.randn(T_, N_) / 10  #
    means = returns.mean(0)
    second_moment_matrix = returns.T @ returns / T_
    covariance_matrix = second_moment_matrix - means.reshape(-1, 1) * means.reshape(1, -1)

    lam = 0.01 * np.exp(np.random.randn(1, N_))

    m_til, m_ = solve_m_tilde_through_iterations(g_bar=means,
                                                 capital_g=second_moment_matrix,
                                                 sigma=covariance_matrix,
                                                 initial_value='high',
                                                 num_iterations=1000,
                                                 lam=lam,
                                                 scale_for_lam=1,
                                                 shrinkage=0)
    print(m_til)

    verify_equation_for_m_and_m_til(lam,
                                    capital_g=second_moment_matrix,
                                    m_=m_,
                                    sigma=covariance_matrix)


if __name__ == '__main__':
    T_ = 1000
    N_ = 3

