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

        if self.fitter == "numpy":
            self.point_estimate = self.estimate()
            if self.n_bootstraps > 0:
                self.vcov = self.bootstrap()
            self.conn.close() if not self.keep_connection_open else None
            return None
        elif self.fitter == "feols":
            fit = self.estimate_feols()
            self.point_estimate = fit.coef().values
            if self.n_bootstraps > 0:
                self.vcov = self.bootstrap()
            fit._vcov = self.vcov
            fit.get_inference()
            fit._vcov_type = "NP-Bootstrap"
            fit._vcov_type_detail = "NP-Bootstrap"
            self.conn.close() if not self.keep_connection_open else None
            return fit

        else:
            raise ValueError(
                "Argument 'fitter' must be 'numpy' or 'feols', got {}".format(
                    self.fitter
                )
            )

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
