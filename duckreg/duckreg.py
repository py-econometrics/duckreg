from abc import ABC, abstractmethod
import duckdb
import numpy as np

######################################################################
class DuckReg(ABC):
    def __init__(
        self, db_name: str, table_name: str, n_bootstraps: int = 100, seed: int = 42
    ):
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.conn = duckdb.connect(db_name)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def compress_data(self):
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
            self.boot_results = self.bootstrap()

    def summary(self):
        if self.n_bootstraps > 0:
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.boot_results)),
            }
        return {"point_estimate": self.point_estimate}


def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    N = np.sqrt(np.diag(n))
    Xn = np.dot(N, X)
    yn = np.dot(y, N)
    betahat = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    return betahat
