"""
Streaming regression leveraging DuckDB's native Arrow IPC support.
"""

import numpy as np
import duckdb
from dataclasses import dataclass
from typing import Optional, Iterator


@dataclass
class RegressionStats:
    """Sufficient statistics for streaming regression."""

    XtX: Optional[np.ndarray] = None
    Xty: Optional[np.ndarray] = None
    yty: float = 0.0
    n: int = 0
    k: Optional[int] = None

    def update(self, X: np.ndarray, y: np.ndarray) -> "RegressionStats":
        """Update statistics with new batch."""
        n_batch, k_batch = X.shape

        if self.XtX is None:
            self.k = k_batch
            self.XtX = np.zeros((k_batch, k_batch))
            self.Xty = np.zeros(k_batch)

        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.yty += y @ y
        self.n += n_batch
        return self

    def solve_ols(self) -> np.ndarray:
        """Compute OLS coefficients."""
        if self.n < self.k:
            return None
        self.check_condition()
        return np.linalg.solve(self.XtX, self.Xty)

    def solve_ridge(self, lambda_: float = 1.0) -> np.ndarray:
        """Compute Ridge coefficients."""
        if self.XtX is None:
            return None
        XtX_reg = self.XtX + lambda_ * np.eye(self.k)
        return np.linalg.solve(XtX_reg, self.Xty)

    def check_condition(self, threshold: float = 1e10):
        """Check the condition number of the XtX matrix."""
        if self.XtX is None:
            return None
        cond = np.linalg.cond(self.XtX)
        if cond > threshold:
            import warnings

            warnings.warn(
                f"High condition number: {cond:.2e}. Consider using Ridge regression."
            )
        return cond


class DuckDBArrowStream:
    """
    Stream data from DuckDB using native Arrow IPC support.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        chunk_size: int = 10000,
        feature_cols: list[str] = None,
        target_col: str = None,
    ):
        self.conn = conn
        self.query = query
        self.chunk_size = chunk_size
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stream data in chunks using DuckDB's Arrow support."""
        result = self.conn.execute(self.query)

        while True:
            arrow_chunk = result.fetch_arrow_table(self.chunk_size)

            if arrow_chunk is None or arrow_chunk.num_rows == 0:
                break

            if self.feature_cols is None:
                self.feature_cols = sorted(
                    [col for col in arrow_chunk.column_names if col.startswith("x")]
                )

            if self.target_col is None:
                self.target_col = "y"

            X = np.column_stack(
                [arrow_chunk[col].to_numpy() for col in self.feature_cols]
            )
            y = arrow_chunk[self.target_col].to_numpy()

            yield (X, y)


class StreamingRegression:
    """
    Streaming regression for duckreg using sufficient statistics.
    Leverages DuckDB's native Arrow IPC support.
    """

    def __init__(
        self, conn: duckdb.DuckDBPyConnection, query: str, chunk_size: int = 10000
    ):
        self.conn = conn
        self.query = query
        self.chunk_size = chunk_size
        self.stats = RegressionStats()

    def fit(self, feature_cols: list[str], target_col: str):
        """
        Perform streaming regression.
        """
        stream = DuckDBArrowStream(
            self.conn, self.query, self.chunk_size, feature_cols, target_col
        )
        for X, y in stream:
            self.stats.update(X, y)
        return self

    def solve_ols(self):
        """
        Solve OLS regression.
        """
        return self.stats.solve_ols()

    def solve_ridge(self, lambda_: float = 1.0):
        """
        Solve Ridge regression.
        """
        return self.stats.solve_ridge(lambda_)

    @classmethod
    def from_table(cls, conn: duckdb.DuckDBPyConnection, table_name: str, **kwargs):
        """Create a StreamingRegression instance from a table name."""
        query = f"SELECT * FROM {table_name}"
        return cls(conn, query, **kwargs)
