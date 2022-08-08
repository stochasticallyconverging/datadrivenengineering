from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import lu_factor, lu_solve


class SVDErrorSimulator:
    def __init__(self, A: np.ndarray,
                 r: int = None,
                 iters: int = 100,
                 randomized: bool = False,
                 svd_method: str = "standard",
                 p: int = 0,
                 q: int = 0,
                 out_bitgen_method: Optional[Callable] = None,
                 eps_bitgen_method: Optional[Callable] = None):
        self._A = A
        if not r:
            r = self._A.shape[1]
        self._U = np.zeros((A.shape[0], r))
        self._S = np.zeros(r)
        self._VT = np.zeros((r, A.shape[0]))
        self._iters = iters
        if randomized:
            self._rsvd(r=r, q=q, p=p)
            self._U, self._S, self._VT = self._U[:, :(1+r)], self._S[:(1+r)], self._VT[:(1+r), :]
        elif svd_method == "snapshot":
            self._snapshot_svd()
            self._U, self._S, self._VT = self._U[:, :(1+r)], self._S[:(1+r)], self._VT[:(1+r), :]
        else:
            self._U, self._S, self._VT = np.linalg.svd(A, full_matrices=False)
            self._U, self._S, self._VT = self._U[:, :(1+r)], self._S[:(1+r)], self._VT[:(1+r), :]
        self.A_reconstructed = self._U @ np.diag(self._S) @ self._VT
        self._errs = np.zeros((iters, 1))
        self._b_mat = None
        self._solved = False
        self.solutions = None
        self._out_bitgen_method = out_bitgen_method if out_bitgen_method else None
        self._eps_bitgen_method = eps_bitgen_method if eps_bitgen_method else self.out_bitgen_method

    def simulate(self, method: str = "lu", **kwargs):
        self._gen_output_mat()
        if method == "lu":
            self.lu_fit()
        elif method == "pinv":
            self.pinv_fit()
        else:
            raise ValueError("User did not choose valid fitting method.")
        self._compute_errs()
        self._plot_error_distribution(**kwargs)

    def lu_fit(self) -> None:
        self.solutions = np.zeros((self._b_mat.shape[0], self._iters))
        lu, piv = lu_factor(self.A_reconstructed)
        for i in range(self._iters):
            self.solutions[:, i] = np.reshape(lu_solve((lu, piv), self._b_mat[:, i]), (-1,))
        self._solved = True

    def pinv_fit(self):
        self.solutions = np.linalg.pinv(self.A_reconstructed) @ self._b_mat
        self._solved = True

    def _compute_errs(self) -> None:
        if self._solved:
            err_sq = np.apply_along_axis(np.linalg.norm, 0, self._b_mat - (self._A @ self.solutions), ord=2)**2
            self._errs = err_sq / np.apply_along_axis(np.linalg.norm, 0, self._b_mat, ord=2)**2
        else:
            raise Exception("The simulator has not solved for x.")

    def _gen_output_mat(self) -> None:
        self._b_mat = np.zeros((self._A.shape[0],))
        self._b_mat[:] = self._out_bitgen_method(size=(self._b_mat.shape[0],))
        self._b_mat = np.repeat(self._b_mat[:, np.newaxis], self._iters, axis=1)
        self._b_mat = self._b_mat + self._eps_bitgen_method(size=self._b_mat.shape)

    def _plot_error_distribution(self, **kwargs):
        if self._solved:
            print(self._errs.shape)
            return plt.hist(self._errs, **kwargs)
        else:
            raise Exception("The simulator has not solved for x.")

    def _rsvd(self, r, q, p) -> None:
        # Step 1: Sample column space of X with P matrix and use pow iter to create a new matrix with rapid sv decay

        ## Step 1a: Sample Column Space of X
        ny = self._A.shape[1]
        P = np.random.randn(ny, r+p)
        Z = self._A @ P

        ## Step 1b:  Power Iterations
        if q:
            for k in range(q):
                Z = self._A @ (self._A.T @ Z)

        ## Step 1c: Obtain orthonormal basis Q for X using QR algorithm
        Q, R = np.linalg.qr(Z, mode="reduced")

        # Step 2: Compute the SVD on projected Y = Q.T @ X

        ## Step 2a: Project X into a smaller space Y
        Y = Q.T @ self._A

        ## Step 2b: Compute SVD on Y
        UY, self._S, self._VT = np.linalg.svd(Y, full_matrices=False)

        ## Step 2c: Construct high-dimensional left singular vectors U
        self._U = Q @ UY

    def _snapshot_svd(self) -> None:
        S, V = np.linalg.eigh(self._A.T @ self._A)
        idx = np.argsort(S)[::-1]
        self._U = self._A @ V[:, idx] @ np.linalg.inv(np.diag(np.sqrt(S[::-1])))
        self._S, self._VT = np.sqrt(S[::-1]), V.T[:, idx]

