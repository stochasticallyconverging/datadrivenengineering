import numpy as np
import matplotlib.pyplot as plt


def plot_distance_from_identity(u: np.ndarray, relative: bool = True) -> None:
    errs = np.zeros((u.shape[1], 1))
    identity = np.eye(u.shape[0], u.shape[1])
    if relative:
        dist = lambda x, idx: np.linalg.norm(identity - (x[:, :idx] @ x[:, :idx].T)) / np.linalg.norm(identity)
    else:
        dist = lambda x, idx: np.linalg.norm(identity - (x[:, :idx] @ x[:, :idx].T))
    for i in range(1, u.shape[1]):
        errs[i] = dist(u, i)
    plt.plot(errs, marker='o', color='k')
    plt.xlabel("r")
    plt.ylabel("Relative Error (Frobenius Norm)")
    plt.title("Distance of U*U.T from Identity as Rank of U_r increases")
    plt.show()


def compute_reconstruction_error(X: np.ndarray,
                                 U: np.ndarray,
                                 S: np.ndarray,
                                 VT: np.ndarray,
                                 squared: bool = True,
                                 relative: bool = True) -> np.ndarray:
    y = np.zeros(VT.shape[1])
    if relative:
        dist = lambda idx: np.linalg.norm(X - (U[:, :idx] @ (np.diag(S[:idx]) @ VT[:idx, :]))) / np.linalg.norm(X)
    else:
        dist = lambda idx: np.linalg.norm(X - (U[:, :idx] @ (np.diag(S[:idx]) @ VT[:idx, :])))

    for i in range(VT.shape[1]):
        y[i] = 1 - dist(i)
        if squared:
            y[i] = y[i]**2

    return y


def plot_reconstruction_metrics_vs_rank(X: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    Uh, Sh, VTh = np.linalg.svd(X, full_matrices=False)
    captured_variance = compute_reconstruction_error(X, Uh, Sh, VTh)
    captured_norm = compute_reconstruction_error(X, Uh, Sh, VTh, squared=False)

    ax[0].plot(captured_variance)
    ax[0].set_title("Captured Variance by Rank")
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("Captured Variance")
    ax[0].axhline(y=0.99)
    ax[0].set_ylim([0.7, 1])
    ax[1].plot(captured_norm)
    ax[1].set_title("Captured Relative Norm by Rank")
    ax[1].set_xlabel("r")
    ax[1].set_xlabel("Captured Error")
    ax[1].axhline(y=0.99)
    ax[1].set_ylim([0.7, 1])
    ax[2].plot(np.cumsum(Sh)/np.sum(Sh))
    ax[2].set_title("Cumulative % Singular Values by Rank")
    ax[2].set_xlabel("r")
    ax[2].set_ylabel("Cumulative Sum")
    ax[2].axhline(y=0.99)
    ax[2].set_ylim([0.7, 1])
