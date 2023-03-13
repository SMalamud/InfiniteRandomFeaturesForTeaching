import numpy as np
from numba import cuda, types, float32, njit, prange


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    TPB = 32

    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x  # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.0
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@njit(parallel=True, cache=True, fastmath=True)
def numba_mat_mul(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros(
        (A.shape[0], B.shape[1]),
    )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
    return res


@njit(parallel=True, cache=True, fastmath=True)
def qr_decomposition(A):
    """Given a square matrix A, returns the QR decomposition"""
    n, m = A.shape
    assert n == m, "A must be a square matrix!"

    # BEGIN Q
    Q = np.empty((n, n))
    u = np.empty((n, n))

    # u1 = v1
    # e1 = u1 / |u1|^2
    # Q = [e1, ..., en]
    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    # u_k
    for k in range(1, n):
        u[:, k] = A[:, k]

        # u_k = v_k - \sum_{j=1}^k (v_k @ uj) * ej
        for j in range(k):
            u[:, k] -= (A[:, k] @ Q[:, j]) * Q[:, j]

        # Q_k = u_k / |u_k|^2
        Q[:, k] = u[:, k] / np.linalg.norm(u[:, k])
    # END Q
    # BEGIN R
    # R is the upper triangular matrix
    # where R_{i,j} = <ei, vj>
    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = Q[:, i] @ A[:, j]
    # END R
    return Q, R


def eig(A, tol=1e-6):
    """
    Finds the eigenvalues and eigenvectors
    of A using QR decomposition
    """

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    while diff > tol:
        A_old[:, :] = A_new
        Q, R = qr_decomposition(A_old)
        A_new[:, :] = R @ Q
        diff = np.abs(A_new - A_old).max()
        i += 1
    eigvals = np.diag(A_new)
    return eigvals
