import time
import torch
import cupy as cp
import numpy as np
from typing import Tuple, Optional


def cp_release_memory_pool():
    """frees GPU memory pools

    See: Memory Pools and
    Cupy Memory Pools for further details
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def cp_tilda_S_k(
    sample_size: int, previous_V: np.ndarray, random_features: np.ndarray
) -> np.ndarray:
    identity = np.eye(sample_size)

    identity_gpu = cp.asarray(identity)
    previous_V_gpu = cp.asarray(previous_V)
    random_features_gpu = cp.asarray(random_features)

    tilda_S_k_gpu = random_features_gpu - previous_V_gpu @ (previous_V_gpu.T @ random_features_gpu)

    # tilda_S_k_gpu = (
    #     identity_gpu - previous_V_gpu @ previous_V_gpu.T
    # ) @ random_features_gpu

    tilda_S_k = cp.asnumpy(tilda_S_k_gpu)
    del identity_gpu
    del previous_V_gpu
    del random_features_gpu
    del tilda_S_k_gpu
    cp_release_memory_pool()
    return tilda_S_k


def cp_eigh(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    start = time.monotonic()
    # GPU acceleration
    covariance_gpu = cp.asarray(X)
    eigval, eigvec1 = cp.linalg.eigh(covariance_gpu)

    end = time.monotonic()
    eig_time = end - start
    print(f"\tEigh Time: {eig_time:.3f}s")
    eigval = cp.asnumpy(eigval)
    eigvec1 = cp.asnumpy(eigvec1)
    return eigval, eigvec1


def cp_three_matrices_multiplication(A: np.ndarray,
                                     B: np.ndarray,
                                     C: np.ndarray,
                                     use_cupy: bool = True,
                                     use_diagonal: bool = True):
    """hard-coded but fast"""
    if use_diagonal:
        # B is diagonal, the B @ C = np.diag(B).reshape(-1, 1) * C. This shoudl be much faster
        C_til = np.diag(B).reshape(-1, 1) * C
    if not use_cupy:
        if use_diagonal:
            return A @ C_til
        return A @ B @ C

    start = time.monotonic()

    if not use_diagonal:
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C_gpu = cp.asarray(C)

        result = A_gpu @ B_gpu @ C_gpu
        end = time.monotonic()
        result_cpu = cp.asnumpy(result)
        del A_gpu
        del B_gpu
        del C_gpu
        del result
    else:
        A_gpu = cp.asarray(A)
        C_til_gpu = cp.asarray(C_til)

        result = A_gpu @ C_til_gpu
        end = time.monotonic()
        result_cpu = cp.asnumpy(result)
        del A_gpu
        del C_til_gpu
        del result

    cp_release_memory_pool()
    final_time = end - start

    return result_cpu


def cp_check_symmetric(a, rtol=1e-05, atol=1e-08):
    return cp.allclose(a, a.T, rtol=rtol, atol=atol)


def cp_rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
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
    x_size = X.nbytes / 1024 / 1024 / 1024
    print(f"Covariance Size: {x_size} GB")
    print(f"GPU Total Space: {torch.cuda.mem_get_info()}")
    print(f"Moving matrix X to GPU")
    print(f"X type: {X.dtype}")
    X_gpu = cp.asarray(X)
    P = cp.random.randn(original_dimensions, r + p)
    print(f"Projecting X to a lower space")
    Z = X_gpu @ P
    # power iterations
    for k in range(q):
        Z = X_gpu @ (X_gpu.T @ Z)
    print(f"Free memory after projection: {torch.cuda.mem_get_info()}")
    Q, R = cp.linalg.qr(Z, mode="reduced")
    del R
    del Z
    cp_release_memory_pool()
    Y = Q.T @ X_gpu
    UY, S, VT = cp.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    return U, S, VT


def cp_hybrid_rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Similar to cp_rSVD

    We compute the projection on CPU but we move
    the QR decomposition and the SVD in GPU
    """

    original_dimensions = X.shape[1]

    # **************************
    # Projection
    # **************************
    print(f"Projecting X with CPU")
    start_proj = time.monotonic()

    P = np.random.randn(original_dimensions, r + p)
    Z = X @ P
    # power iterations
    for _ in range(q):
        Z = X @ (X.T @ Z)
    end_proj = time.monotonic()
    proj_time = end_proj - start_proj
    print(f"CPU Projection time: {proj_time:.3f}")

    # *****************************
    # QR and SVD
    # *****************************
    print(f"Z size: {Z.nbytes / 1024/ 1024/1024}Gb")
    start_time = time.monotonic()
    Z_gpu = cp.asarray(Z)
    Q, R = cp.linalg.qr(Z_gpu, mode="reduced")

    # Free GPU memory
    del R
    del Z_gpu
    cp_release_memory_pool()
    Y_cpu = cp.asnumpy(Q.T) @ X
    Y = cp.asarray(Y_cpu)
    UY, S, VT = cp.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    end_time = time.monotonic()
    decomposition_time = end_time - start_time
    print(f"QR and SVD decomposition took: {decomposition_time:.3f}s")
    return U, S, VT


def cp_eigh_from_rSVD(
    X: np.ndarray, r: int, q: int, p: int, hybrid: Optional[bool] = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    """returns sorted eigenvalues and eigenvectors from rSVD"""
    start = time.monotonic()
    if hybrid:
        U, S, VT = cp_hybrid_rSVD(X, r, q, p)
    else:
        U, S, VT = cp_rSVD(X, r, q, p)
    del VT
    cp_release_memory_pool()
    eigenvalues = cp.flip(S)
    eigenvectors = cp.flip(U, axis=1)
    end = time.monotonic()
    eigh_time = end - start
    print(f"GPU Eigenvalue rSVD time: {eigh_time:.3f}")
    return eigenvalues, eigenvectors


def cp_hybrid_eigh_from_rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """returns sorted eigenvalues and eigenvectors from rSVD using both CPU and GPU"""
    start = time.monotonic()
    U, S, VT = cp_hybrid_rSVD(X, r, q, p)

    eigenvalues = cp.asnumpy(cp.flip(S))
    eigenvectors = cp.asnumpy(cp.flip(U, axis=1))
    end = time.monotonic()
    eigh_time = end - start
    print(f"GPU Eigenvalue rSVD time: {eigh_time:.3f}")
    del VT
    del S
    del U
    cp_release_memory_pool()
    return eigenvalues, eigenvectors
