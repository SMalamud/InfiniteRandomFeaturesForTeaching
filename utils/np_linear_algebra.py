import numpy as np

from typing import Tuple


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """computes the randomized SVD decomposition

    Args:
        X, the matrix to be decomposed
        r, projection to a smaller space
        q, power iteration
        p, oversampling


    credits to:
    prof. Steve Brunton http://databookuw.com/page-2/page-4/
    """

    original_dimensions = X.shape[1]
    P = np.random.randn(original_dimensions, r + p)
    Z = X @ P
    # power iterations
    for k in range(q):
        Z = X @ (X.T @ Z)
    print(Z.shape)
    Q, R = np.linalg.qr(Z, mode="reduced")

    Y = Q.T @ X

    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    return U, S, VT
