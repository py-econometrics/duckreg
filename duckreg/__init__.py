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
from .bitpacking import (
    BitmapMundlak,
    BitmapMundlakEventStudy,
    BitmapPanel,
    LongPanelBitPacker,
)

__all__ = [
    "BitmapMundlak",
    "BitmapMundlakEventStudy",
    "BitmapPanel",
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
    "LongPanelBitPacker",
    "ridge_closed_form",
    "ridge_closed_form_batch",
    "wls",
]
