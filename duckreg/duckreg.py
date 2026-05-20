from abc import ABC, abstractmethod
from typing import Any

import ibis
import numpy as np
import pandas as pd


def _looks_like_ibis_backend(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ("execute", "sql", "table"))


def _is_select_query(query: str) -> bool:
    query_start = query.lstrip().lower()
    return query_start.startswith("select") or query_start.startswith("with")


class IbisQueryResult:
    """Small result wrapper matching the DuckDB calls used by duckreg."""

    def __init__(self, dataframe: pd.DataFrame | None = None, cursor: Any = None):
        self._dataframe = dataframe
        self._cursor = cursor

    def _materialize(self) -> pd.DataFrame:
        if self._dataframe is not None:
            return self._dataframe

        if hasattr(self._cursor, "fetchdf"):
            self._dataframe = self._cursor.fetchdf()
            return self._dataframe

        if hasattr(self._cursor, "df"):
            self._dataframe = self._cursor.df()
            return self._dataframe

        rows = self._cursor.fetchall()
        description = getattr(self._cursor, "description", None) or []
        columns = [col[0] for col in description] or None
        self._dataframe = pd.DataFrame(rows, columns=columns)
        return self._dataframe

    def fetchdf(self) -> pd.DataFrame:
        return self._materialize()

    def df(self) -> pd.DataFrame:
        return self.fetchdf()

    def fetchall(self) -> list[tuple]:
        df = self._materialize()
        return list(df.itertuples(index=False, name=None))

    def fetchone(self) -> tuple | None:
        rows = self.fetchall()
        return rows[0] if rows else None


class IbisConnectionAdapter:
    """DuckDB-like execution facade over an Ibis backend."""

    def __init__(self, backend: Any):
        self.backend = backend

    @property
    def backend_name(self) -> str:
        module_parts = type(self.backend).__module__.split(".")
        if len(module_parts) >= 3 and module_parts[:2] == ["ibis", "backends"]:
            return module_parts[2]
        return type(self.backend).__name__

    def execute(self, query: str, params: list | tuple | None = None) -> IbisQueryResult:
        if params is None and _is_select_query(query):
            dataframe = self.backend.sql(query).execute(limit=None)
            if isinstance(dataframe, pd.Series):
                dataframe = dataframe.to_frame()
            elif not isinstance(dataframe, pd.DataFrame):
                dataframe = pd.DataFrame([[dataframe]])
            return IbisQueryResult(dataframe=dataframe)

        kwargs = {"parameters": params} if params is not None else {}
        cursor = self.backend.raw_sql(query, **kwargs)
        return IbisQueryResult(cursor=cursor)

    def table(self, table_name: str):
        try:
            return self.backend.table(table_name)
        except Exception:
            return self.backend.sql(f"SELECT * FROM {table_name}")

    def to_pandas(self, expr) -> pd.DataFrame:
        return self.backend.to_pandas(expr, limit=None)

    def close(self):
        if hasattr(self.backend, "disconnect"):
            self.backend.disconnect()


def _connect_ibis_backend(
    db_name: str | None, connection: Any = None
) -> tuple[Any, bool]:
    if (
        connection is not None
        and db_name is not None
        and _looks_like_ibis_backend(db_name)
    ):
        raise ValueError(
            "Pass an Ibis backend through either db_name or connection, not both"
        )

    if connection is None and _looks_like_ibis_backend(db_name):
        return db_name, False

    if connection is not None:
        if isinstance(connection, str):
            return ibis.connect(connection), True
        if not _looks_like_ibis_backend(connection):
            raise TypeError(
                "connection must be an Ibis backend or an Ibis connection URL"
            )
        return connection, False

    if db_name is not None and "://" in db_name:
        return ibis.connect(db_name), True

    database = ":memory:" if db_name is None else db_name
    try:
        return ibis.duckdb.connect(database), True
    except ImportError as exc:
        raise ImportError(
            "DuckDB-style db_name connections require the Ibis DuckDB backend. "
            "Install duckreg with the duckdb extra or pass an explicit Ibis backend."
        ) from exc


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
        connection=None,
    ):
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        backend, owns_connection = _connect_ibis_backend(db_name, connection)
        self.conn = IbisConnectionAdapter(backend)
        self.backend = backend
        self._owns_connection = owns_connection
        self.rng = np.random.default_rng(seed)
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open

    def _close_connection(self):
        if self._owns_connection and not self.keep_connection_open:
            self.conn.close()

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
        self._close_connection()
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
