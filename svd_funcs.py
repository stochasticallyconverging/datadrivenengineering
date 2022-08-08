import math
import numpy as np
import scipy as sp


def truncation_coef(n: float, m: float, gamma: float = 1) -> float:
    beta = float(m)/float(n)
    return (2*(beta + 1) + ((8*beta)/(beta + 1) + math.sqrt(beta**2 + 14*beta + 1)))*math.sqrt(n)*gamma

def is_diag(S):
    i, j = S.shape
    assert i == j
    test = S.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])


def rSVD(X, r, q, p):
    # Step 1: Sample column space of X with P matrix and use pow iter to create a new matrix with more rapid sv decay

    ## Step 1a: Sample Column Space of X
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P

    ## Step 1b:  Power Iterations
    for k in range(q):
        Z = X @ (X.T @ Z)

    ## Step 1c: Obtain orthonormal basis Q for X using QR algorithm
    Q, R = np.linalg.qr(Z, mode="reduced")

    # Step 2: Compute the SVD on projected Y = Q.T @ X

    ## Step 2a: Project X into a smaller space Y
    Y = Q.T @ X

    ## Step 2b: Compute SVD on Y
    UY, S, VT = np.linalg.svd(Y, full_matrices=False)

    ## Step 2c: Construct high-dimensional left singular vectors U
    U = Q @ UY

    return U, S, VT


def svd_solve(U: np.ndarray, S: np.ndarray, VT: np.ndarray, b, r: int = None, *args, **kwargs):
    if not r:
        r = U.shape[0]
    if not is_diag(S):
        S = np.diag(S)
    return sp.linalg.solve(U[:, :(1+r)] @ S[:(r+1), :(r+1)] @ VT[:(r+1), :], b, *args, **kwargs)

def iterative_lu_solve()