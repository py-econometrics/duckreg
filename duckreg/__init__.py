"""
.. include:: ../README.md
"""

from .duckreg import DuckReg, ridge_closed_form, ridge_closed_form_batch, wls
from .dbreg import (
    DBDML,
    DBDoubleDemeaning,
    DBMundlak,
    DBReg,
    DBRegression,
    DbDML,
    DbDoubleDemeaning,
    DbMundlak,
    DbReg,
    DbRegression,
)
from .estimators import (
    DuckDML,
    DuckDoubleDemeaning,
    DuckLogisticRegression,
    DuckMultinomialLogisticRegression,
    DuckMundlak,
    DuckMundlakEventStudy,
    DuckPoissonMultinomialRegression,
    DuckPoissonRegression,
    DuckRegression,
)
from .regularized import DuckRidge

__all__ = [
    "DBDML",
    "DBDoubleDemeaning",
    "DBMundlak",
    "DBReg",
    "DBRegression",
    "DbDML",
    "DbDoubleDemeaning",
    "DbMundlak",
    "DbReg",
    "DbRegression",
    "DuckDML",
    "DuckDoubleDemeaning",
    "DuckLogisticRegression",
    "DuckMultinomialLogisticRegression",
    "DuckMundlak",
    "DuckMundlakEventStudy",
    "DuckPoissonMultinomialRegression",
    "DuckPoissonRegression",
    "DuckReg",
    "DuckRegression",
    "DuckRidge",
    "ridge_closed_form",
    "ridge_closed_form_batch",
    "wls",
]
