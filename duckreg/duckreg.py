from abc import ABC, abstractmethod
import duckdb
import numpy as np


######################################################################
class DuckReg(ABC):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        n_bootstraps: int = 100,
        fitter="numpy",
        keep_connection_open=False,
    ):
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.conn = duckdb.connect(db_name)
        self.rng = np.random.default_rng(seed)
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def compress_data(self):
        pass

    @abstractmethod
    def collect_data(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def bootstrap(self):
        pass

    def fit(self):
        self.prepare_data()
        self.compress_data()

        self.point_estimate = self.estimate()
        if self.n_bootstraps > 0:
            self.vcov = self.bootstrap()
        self.conn.close() if not self.keep_connection_open else None
        return None

    def summary(self) -> dict:
        """Summary of regression

        Returns:
            dict
        """
        if self.n_bootstraps > 0:
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}

    def queries(self) -> dict:
        """Collect all query methods in the class

        Returns:
            dict: Dictionary of query methods
        """
        self._query_names = [x for x in dir(self) if "query" in x]
        self.queries = {
            k: getattr(self, self._query_names[c])
            for c, k in enumerate(self._query_names)
        }
        return self.queries


def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Weighted least squares with frequency weights"""
    N = np.sqrt(n)
    N = N.reshape(-1, 1) if N.ndim == 1 else N
    Xn = X * N
    yn = y * N
    betahat = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    return betahat


def ridge_closed_form(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float
) -> np.ndarray:
    """Ridge regression with data augmented representation
    Trad ridge: (X'X + lam I)^{-1} X' y
    Augmentation: Xtilde = [X; sqrt(lam) I], ytilde = [y; 0]
    this lets us use lstsq solver, which is more optimized than using normal equations

    Args:
        X (np.ndarray): Design matrix
        y (np.ndarray): Outcome vector
        n (np.ndarray): Frequency weights
        lam (float): Regularization parameter

    Returns:
        np.ndarray: Coefficient estimates
    """
    k = X.shape[1]
    N = np.sqrt(n)
    Xn = X * N
    yn = y * N
    Xtilde = np.r_[Xn, np.diag(np.repeat(np.sqrt(lam), k))]
    ytilde = np.concatenate([yn, np.zeros(shape=(k, 1))])
    betahat = np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0]
    return betahat


def ridge_closed_form_batch(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, lambda_grid: np.ndarray
) -> np.ndarray:
    """Optimized ridge regression for multiple lambda values
    Pre-computes reusable components to avoid repeated work in lambda grid search

    Args:
        X (np.ndarray): Design matrix
        y (np.ndarray): Outcome vector
        n (np.ndarray): Frequency weights
        lambda_grid (np.ndarray): Array of regularization parameters

    Returns:
        np.ndarray: Coefficient estimates, shape (n_lambdas, n_features)
    """
    k = X.shape[1]
    n_lambdas = len(lambda_grid)

    # Pre-compute weight matrix (done once)
    N = np.sqrt(n)
    # Pre-compute weighted X and y (done once)
    Xn = X * N
    yn = y * N

    # Pre-allocate identity matrix and zero vector (done once)
    I_k = np.eye(k)
    zeros_k = np.zeros((k, 1))

    # Pre-allocate result array
    coefs = np.zeros((n_lambdas, k))

    # Loop over lambda values (only lambda-dependent operations)
    for i, lam in enumerate(lambda_grid):
        # Only lambda-dependent work: scale identity and concatenate
        sqrt_lam_I = np.sqrt(lam) * I_k
        Xtilde = np.vstack([Xn, sqrt_lam_I])
        ytilde = np.vstack([yn, zeros_k])

        coefs[i, :] = np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0].flatten()

    return coefs
