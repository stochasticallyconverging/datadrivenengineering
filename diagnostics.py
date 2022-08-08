import numpy as np


def gen_svd_problem(U, S, VT, iters):
    for i in range(iters):
        b = np.random.rand(U.shape[0], 1)
        eps = np.random.rand(U.shape[0], iters)
        yield U @ np.diag(S) @ VT, b + eps, np.repeat()


def compute_relative_error(out_true: np.ndarray, input: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(out_true - A @ input))**2 / np.linalg.norm(out_true)**2
