"""
.. include:: ../README.md
"""

from .duckreg import DuckReg, ridge_closed_form, ridge_closed_form_batch, wls
from .dbreg import (
    DBDML,
    DBDoubleDemeaning,
    DBLogisticRegression,
    DBMundlak,
    DBMundlakEventStudy,
    DBMultinomialLogisticRegression,
    DBPoissonMultinomialRegression,
    DBPoissonRegression,
    DBReg,
    DBRegression,
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
    "DBLogisticRegression",
    "DBMundlak",
    "DBMundlakEventStudy",
    "DBMultinomialLogisticRegression",
    "DBPoissonMultinomialRegression",
    "DBPoissonRegression",
    "DBReg",
    "DBRegression",
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
