import time
import torch
import numpy as np
from typing import Tuple, Optional


def torch_check_symmetric(a, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)


def torch_rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """computes the randomized SVD decomposition

    Args:
        X, the matrix to be decomposed
        r, projection to a smaller space
        q, power iteration
        p, oversampling


    credits to:
    prof. Steve Brunton http://databookuw.com/page-2/page-4/
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    original_dimensions = X.shape[1]
    x_size = X.nbytes / 1024 / 1024 / 1024
    print(f"Covariance Size: {x_size} GB")
    print(f"GPU Total Space: {torch.cuda.mem_get_info()}")
    print(f"Moving matrix X to GPU")
    print(f"X type: {X.dtype}")
    # For big matrixes this will likely fill one entire GPU
    X_gpu = torch.Tensor(X).to(device)

    P = torch.Tensor(np.random.randn(original_dimensions, r + p)).to(device)
    print(f"Projecting X to a lower space")

    Z = torch.matmul(X_gpu, P).to(device)
    # power iterations
    # We need to do multigpu matmul
    for k in range(q):
        Z = torch.matmul(X_gpu, (torch.matmul(X_gpu.T, Z)))
    print(f"Free memory after projection: {torch.cuda.mem_get_info()}")
    Q, R = torch.linalg.qr(Z, mode="reduced")
    del R
    del Z
    torch.cuda.empty_cache()

    print(f"Moving Y and Q to GPU 2")
    
    # We can kill X_gpu after the following line
    Y = torch.matmul(Q.T, X_gpu).to("cuda:1")
    Q = Q.to("cuda:1")
    
    UY, S, VT = torch.linalg.svd(Y, full_matrices=False)
    del Y, X_gpu
    torch.cuda.empty_cache()
    U = torch.matmul(Q, UY)
    return U, S, VT


def torch_eigh_from_rSVD(
    X: np.ndarray, r: int, q: int, p: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """returns sorted eigenvalues and eigenvectors from rSVD"""
    start = time.monotonic()
    U, S, VT = torch_rSVD(X, r, q, p)

    end = time.monotonic()
    eigh_time = end - start
    eigenvalues = np.flip(S.cpu().numpy())
    eigenvectors = torch.fliplr(U).cpu().numpy()
    del VT, S, U
    torch.cuda.empty_cache()
    print(f"GPU Eigenvalue rSVD time: {eigh_time:.3f}")
    return eigenvalues, eigenvectors
