import numpy as np
import pandas as pd
from tqdm import tqdm
from .duckreg import DuckReg, ridge_closed_form, ridge_closed_form_batch


class DuckRidge(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        formula: str,
        lambda_grid=None,
        cv_folds: int = 5,
        seed: int = 42,
        n_bootstraps: int = 0,  # Disable by default - ridge SEs are complex
        rowid_col: str = "rowid",
        fitter: str = "ridge",
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            fitter=fitter,
            **kwargs,
        )
        self.formula = formula
        self.lambda_grid = (
            lambda_grid if lambda_grid is not None else np.logspace(-4, 2, 50)
        )
        self.cv_folds = cv_folds
        self.rowid_col = rowid_col
        self.best_lambda = None
        self.cv_scores = None
        self.lambda_path_coefs = None
        self._parse_formula()

    def _parse_formula(self):
        """Parse formula similar to DuckRegression"""
        lhs, rhs = self.formula.split("~")
        rhs_deparsed = rhs.split("|")
        covars, fevars = rhs.split("|") if len(rhs_deparsed) > 1 else (rhs, None)

        self.outcome_vars = [x.strip() for x in lhs.split("+")]
        self.covars = [x.strip() for x in covars.split("+")]
        self.fevars = [x.strip() for x in fevars.split("+")] if fevars else []
        self.strata_cols = self.covars + self.fevars

        if len(self.outcome_vars) > 1:
            raise ValueError(
                "DuckRidge currently supports single outcome variable only"
            )

        if not self.outcome_vars:
            raise ValueError("No outcome variables found in the formula")

    def prepare_data(self):
        """Prepare CV fold assignments if needed"""
        if self.cv_folds > 1:
            self._prepare_cv_folds()

    def _prepare_cv_folds(self):
        """Add fold_id column to table for cross-validation"""
        # Create fold assignments using row_number() for reproducible folds
        fold_query = f"""
        CREATE TEMP TABLE cv_folds AS
        SELECT *,
               (ROW_NUMBER() OVER (ORDER BY {self.rowid_col})) % {self.cv_folds} AS fold_id
        FROM {self.table_name}
        """
        self.conn.execute(fold_query)
        # Update table_name to use the temp table with fold assignments
        self.table_name = "cv_folds"

    def compress_data(self):
        """Compress data similar to DuckRegression"""
        # Build GROUP BY columns
        group_by_cols = ", ".join(self.strata_cols)

        # Add fold_id to grouping if doing CV
        if self.cv_folds > 1:
            group_by_cols += ", fold_id"
            select_cols = ", ".join(self.strata_cols) + ", fold_id"
        else:
            select_cols = ", ".join(self.strata_cols)

        # Build aggregation expressions
        outcome_var = self.outcome_vars[0]
        agg_expressions = [
            "COUNT(*) as count",
            f"SUM({outcome_var}) as sum_{outcome_var}",
            f"SUM(POW({outcome_var}, 2)) as sum_{outcome_var}_sq",
        ]

        all_agg_expressions = ", ".join(agg_expressions)

        self.compress_query = f"""
        SELECT {select_cols}, {all_agg_expressions}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """

        self.df_compressed = self.conn.execute(self.compress_query).fetchdf()

        # Add mean column
        mean_col = f"mean_{outcome_var}"
        self.df_compressed[mean_col] = (
            self.df_compressed[f"sum_{outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame, include_intercept: bool = True):
        """Collect X, y, n from compressed data"""
        outcome_var = self.outcome_vars[0]
        y = data[f"mean_{outcome_var}"].values
        X = data[self.covars].values
        n = data["count"].values

        # Ensure proper dimensions
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        n = n.reshape(-1, 1) if n.ndim == 1 else n

        # Add intercept if no fixed effects and requested
        if include_intercept and not self.fevars:
            X = np.c_[np.ones(X.shape[0]), X]

        # Handle fixed effects (simplified - could expand this)
        if self.fevars:
            raise NotImplementedError("Fixed effects not yet supported in DuckRidge")

        return y, X, n

    def estimate(self, lam: float = 0.1):
        """Estimate ridge regression for single lambda"""
        y, X, n = self.collect_data(self.df_compressed)
        betahat = ridge_closed_form(X, y, n, lam)
        return betahat.flatten()

    def fit_lambda_path(self):
        """Fit ridge regression across lambda grid using optimized batch function"""
        y, X, n = self.collect_data(self.df_compressed)

        # Use optimized batch function
        self.lambda_path_coefs = ridge_closed_form_batch(X, y, n, self.lambda_grid)
        return self.lambda_path_coefs

    def cross_validate(self):
        """Perform k-fold cross-validation to select best lambda"""
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be > 1 for cross-validation")

        cv_errors = np.zeros((len(self.lambda_grid), self.cv_folds))

        for fold in range(self.cv_folds):
            # Split data by fold
            train_data = self.df_compressed[self.df_compressed["fold_id"] != fold]
            test_data = self.df_compressed[self.df_compressed["fold_id"] == fold]

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # Get train and test sets
            y_train, X_train, n_train = self.collect_data(train_data)
            y_test, X_test, n_test = self.collect_data(test_data)

            # Fit for each lambda and compute test error
            for i, lam in enumerate(self.lambda_grid):
                # Fit on training data
                beta_hat = ridge_closed_form(X_train, y_train, n_train, lam)

                # Predict on test data
                y_pred = X_test @ beta_hat

                # Weighted MSE
                mse = np.sum(
                    n_test * (y_test.flatten() - y_pred.flatten()) ** 2
                ) / np.sum(n_test)
                cv_errors[i, fold] = mse

        # Average CV errors across folds
        self.cv_scores = np.mean(cv_errors, axis=1)

        # Select best lambda
        best_idx = np.argmin(self.cv_scores)
        self.best_lambda = self.lambda_grid[best_idx]

        return self.best_lambda, self.cv_scores

    def fit(self, lambda_selection="cv"):
        """Main fit method"""
        # Standard duckreg preparation and compression
        self.prepare_data()
        self.compress_data()

        if lambda_selection == "cv" and self.cv_folds > 1:
            # Cross-validation to select lambda
            self.cross_validate()
            self.point_estimate = self.estimate(self.best_lambda)
        elif lambda_selection == "path":
            # Fit entire lambda path
            self.fit_lambda_path()
            # Use middle lambda as default
            mid_idx = len(self.lambda_grid) // 2
            self.point_estimate = self.lambda_path_coefs[mid_idx, :]
        else:
            # Single lambda (use first in grid or default)
            single_lambda = self.lambda_grid[0] if hasattr(self, "lambda_grid") else 0.1
            self.point_estimate = self.estimate(single_lambda)

        # Close connection if not keeping open
        if not self.keep_connection_open:
            self.conn.close()

        return None

    def bootstrap(self):
        """Bootstrap for ridge - placeholder (complex due to bias correction)"""
        raise NotImplementedError(
            "Bootstrap SEs not yet implemented for ridge regression"
        )

    def summary(self) -> dict:
        """Summary of ridge regression results"""
        result = {"point_estimate": self.point_estimate}

        if hasattr(self, "best_lambda") and self.best_lambda is not None:
            result["best_lambda"] = self.best_lambda

        if hasattr(self, "cv_scores") and self.cv_scores is not None:
            result["cv_scores"] = self.cv_scores
            result["lambda_grid"] = self.lambda_grid

        if hasattr(self, "lambda_path_coefs") and self.lambda_path_coefs is not None:
            result["lambda_path_coefs"] = self.lambda_path_coefs
            result["lambda_grid"] = self.lambda_grid

        return result

    def plot_cv_curve(self):
        """Plot cross-validation curve (requires matplotlib)"""
        if not hasattr(self, "cv_scores") or self.cv_scores is None:
            raise ValueError("Must run cross-validation first")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.semilogx(self.lambda_grid, self.cv_scores, "bo-")
            plt.axvline(
                self.best_lambda,
                color="red",
                linestyle="--",
                label=f"Best λ = {self.best_lambda:.4f}",
            )
            plt.xlabel("Regularization parameter (λ)")
            plt.ylabel("Cross-validation error")
            plt.title("Ridge Regression Cross-Validation")
            plt.legend()
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib not available - cannot plot CV curve")

    def plot_coefficient_path(self):
        """Plot coefficient paths across lambda values"""
        if not hasattr(self, "lambda_path_coefs") or self.lambda_path_coefs is None:
            raise ValueError("Must fit lambda path first")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            for i in range(self.lambda_path_coefs.shape[1]):
                plt.semilogx(
                    self.lambda_grid, self.lambda_path_coefs[:, i], label=f"β_{i}"
                )

            if hasattr(self, "best_lambda") and self.best_lambda is not None:
                plt.axvline(
                    self.best_lambda,
                    color="red",
                    linestyle="--",
                    label=f"Best λ = {self.best_lambda:.4f}",
                )

            plt.xlabel("Regularization parameter (λ)")
            plt.ylabel("Coefficient value")
            plt.title("Ridge Regression Coefficient Paths")
            plt.legend()
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib not available - cannot plot coefficient paths")
