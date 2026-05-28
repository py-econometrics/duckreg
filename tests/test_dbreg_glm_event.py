import ibis
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import duckreg
from duckreg.dbreg import (
    DBLogisticRegression,
    DBMultinomialLogisticRegression,
    DBMundlakEventStudy,
    DBPoissonMultinomialRegression,
    DBPoissonRegression,
)
from duckreg.estimators import (
    DuckLogisticRegression,
    DuckMultinomialLogisticRegression,
    DuckMundlakEventStudy,
    DuckPoissonMultinomialRegression,
    DuckPoissonRegression,
)


def _make_con(tmp_path, df, table="data"):
    con = ibis.duckdb.connect(tmp_path / "dbreg_glm_event.db")
    con.create_table(table, df, overwrite=True)
    return con


def _add_intercept(values):
    return np.c_[np.ones(len(values)), values]


def _fit_logistic_mle(X, y):
    def objective(beta):
        eta = X @ beta
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = X.T @ (1 / (1 + np.exp(-eta)) - y)
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(X.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )
    assert result.success
    return result.x


def _fit_poisson_mle(X, y):
    def objective(beta):
        eta = X @ beta
        mu = np.exp(eta)
        value = np.sum(mu - y * eta)
        gradient = X.T @ (mu - y)
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(X.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )
    assert result.success
    return result.x


def _fit_multinomial_mle(X, y_codes, n_labels):
    n_nonbase = n_labels - 1
    n_features = X.shape[1]

    def objective(flat_beta):
        beta = flat_beta.reshape(n_nonbase, n_features)
        eta = X @ beta.T
        full_eta = np.c_[eta, np.zeros(len(X))]
        log_denom = logsumexp(full_eta, axis=1)
        value = np.sum(log_denom - full_eta[np.arange(len(X)), y_codes])
        probs = np.exp(full_eta - log_denom[:, None])[:, :n_nonbase]
        indicators = np.column_stack([y_codes == j for j in range(n_nonbase)])
        gradient = -((indicators - probs).T @ X).reshape(-1)
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(n_nonbase * n_features),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )
    assert result.success
    return result.x.reshape(n_nonbase, n_features)


def test_db_glm_exports():
    assert duckreg.DBLogisticRegression is DBLogisticRegression
    assert duckreg.DBPoissonRegression is DBPoissonRegression
    assert duckreg.DBMultinomialLogisticRegression is DBMultinomialLogisticRegression
    assert duckreg.DBPoissonMultinomialRegression is DBPoissonMultinomialRegression
    assert duckreg.DBMundlakEventStudy is DBMundlakEventStudy


def test_db_logistic_matches_duck_logistic(tmp_path):
    rng = np.random.default_rng(123)
    n = 5000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    eta = -0.7 + 0.45 * x1 - 0.35 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=123,
        method="irls",
        n_bootstraps=0,
    )
    old = DuckLogisticRegression(**kwargs)
    new = DBLogisticRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_logistic_matches_scipy_log_likelihood(tmp_path):
    rng = np.random.default_rng(124)
    n = 3000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    eta = -0.6 + 0.35 * x1 - 0.25 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    con = _make_con(tmp_path, df)
    model = DBLogisticRegression(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=124,
        method="irls",
        n_bootstraps=0,
    )
    model.fit()
    expected = _fit_logistic_mle(_add_intercept(df[["x1", "x2"]].values), y)
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-7, atol=1e-7)


def test_db_poisson_matches_duck_poisson(tmp_path):
    rng = np.random.default_rng(456)
    n = 5000
    x1 = rng.integers(0, 5, n)
    x2 = rng.integers(0, 2, n)
    mu = np.exp(0.2 + 0.18 * x1 - 0.25 * x2)
    y = rng.poisson(mu)
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=456,
        method="irls",
        n_bootstraps=0,
    )
    old = DuckPoissonRegression(**kwargs)
    new = DBPoissonRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_poisson_matches_scipy_log_likelihood(tmp_path):
    rng = np.random.default_rng(457)
    n = 3000
    x1 = rng.integers(0, 5, n)
    x2 = rng.integers(0, 2, n)
    mu = np.exp(0.15 + 0.12 * x1 - 0.22 * x2)
    y = rng.poisson(mu)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    con = _make_con(tmp_path, df)
    model = DBPoissonRegression(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=457,
        method="irls",
        n_bootstraps=0,
    )
    model.fit()
    expected = _fit_poisson_mle(_add_intercept(df[["x1", "x2"]].values), y)
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-7, atol=1e-7)


def test_db_multinomial_logit_matches_duck_multinomial_logit(tmp_path):
    rng = np.random.default_rng(789)
    n = 4000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    X = np.c_[np.ones(n), x1, x2]
    beta = np.array([[0.2, 0.35, -0.2], [-0.4, -0.15, 0.3]])
    eta = np.c_[X @ beta.T, np.zeros(n)]
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(["a", "b", "c"], p=prob) for prob in probs])
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=789,
        labels=["a", "b", "c"],
        baseline="c",
        n_bootstraps=0,
    )
    old = DuckMultinomialLogisticRegression(**kwargs)
    new = DBMultinomialLogisticRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_multinomial_logit_matches_scipy_log_likelihood(tmp_path):
    rng = np.random.default_rng(790)
    n = 3000
    labels = ["a", "b", "c"]
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    X = _add_intercept(np.c_[x1, x2])
    beta = np.array([[0.25, 0.28, -0.14], [-0.35, -0.1, 0.22]])
    eta = np.c_[X @ beta.T, np.zeros(n)]
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_codes = np.array([rng.choice(len(labels), p=prob) for prob in probs])
    y = np.array(labels)[y_codes]
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    con = _make_con(tmp_path, df)
    model = DBMultinomialLogisticRegression(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=790,
        labels=labels,
        baseline="c",
        n_bootstraps=0,
    )
    model.fit()
    expected = _fit_multinomial_mle(X, y_codes, len(labels))
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-7, atol=1e-7)


def test_db_poisson_multinomial_matches_duck_poisson_multinomial(tmp_path):
    rng = np.random.default_rng(321)
    labels = [f"label_{j}" for j in range(8)]
    rows = []
    for i in range(250):
        x1 = rng.integers(0, 4)
        x2 = rng.integers(0, 2)
        for j, label in enumerate(labels):
            mu = np.exp(-1.0 + 0.08 * j + 0.15 * x1 - 0.1 * x2)
            rows.append((i, label, rng.poisson(mu), x1, x2))
    df = pd.DataFrame(rows, columns=["row_id", "label", "count", "x1", "x2"])
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        count_col="count",
        label_col="label",
        covars=["x1", "x2"],
        seed=321,
        n_bootstraps=0,
    )
    old = DuckPoissonMultinomialRegression(**kwargs)
    new = DBPoissonMultinomialRegression(**kwargs)
    old.fit()
    new.fit()
    pd.testing.assert_frame_equal(
        new.summary()["point_estimate"],
        old.summary()["point_estimate"],
        check_exact=False,
    )


def test_db_event_study_matches_duck_event_study(tmp_path):
    rows = []
    for unit, cohort in [(1, 2), (2, 3), (3, None), (4, 2), (5, None)]:
        for time in range(5):
            treated = int(cohort is not None and time >= cohort)
            y = 1.0 + 0.2 * unit + 0.3 * time + 0.8 * treated
            rows.append((unit, time, treated, y, unit))
    df = pd.DataFrame(rows, columns=["unit", "time", "D", "Y", "cluster"])
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_col="D",
        unit_col="unit",
        time_col="time",
        cluster_col="cluster",
        seed=42,
        n_bootstraps=0,
    )
    old = DuckMundlakEventStudy(**kwargs)
    new = DBMundlakEventStudy(**kwargs)
    old.fit()
    new.fit()

    old_est = old.summary()["point_estimate"]
    new_est = new.summary()["point_estimate"]
    assert new_est.keys() == old_est.keys()
    for cohort in old_est:
        pd.testing.assert_frame_equal(new_est[cohort], old_est[cohort], check_exact=False)

    for estimator in (old, new):
        assert "treatment_time_2_1" not in estimator.rhs_cols
        assert "treatment_time_3_2" not in estimator.rhs_cols

    assert old_est["2"].loc["treatment_time_2_1", "est"] == 0.0
    assert old_est["3"].loc["treatment_time_3_2", "est"] == 0.0


def test_db_event_study_ibis_materialization_omits_reference_periods(tmp_path):
    rows = []
    for unit, cohort in [(1, 2), (2, 3), (3, None), (4, 2), (5, None), (6, 3)]:
        for time in range(5):
            treated = int(cohort is not None and time >= cohort)
            y = 1.0 + 0.2 * unit + 0.3 * time + 0.8 * treated
            rows.append((unit, time, treated, y, unit))
    df = pd.DataFrame(rows, columns=["unit", "time", "D", "Y", "cluster"])
    con = _make_con(tmp_path, df)

    model = DBMundlakEventStudy(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_col="D",
        unit_col="unit",
        time_col="time",
        cluster_col="cluster",
        seed=43,
        n_bootstraps=0,
        pre_treat_interactions=True,
    )
    model.prepare_data()

    assert isinstance(model.design_matrix, ibis.expr.types.Table)
    assert "treatment_time_2_1" not in model.design_matrix.columns
    assert "treatment_time_3_2" not in model.design_matrix.columns
    assert "treatment_time_2_0" in model.design_matrix.columns
    assert "treatment_time_3_1" in model.design_matrix.columns

    model.compress_data()
    assert "treatment_time_2_1" not in model.rhs_cols
    assert "treatment_time_3_2" not in model.rhs_cols
    assert all("treatment_time_2_1" not in col for col in model.df_compressed.columns)
    assert all("treatment_time_3_2" not in col for col in model.df_compressed.columns)

    model.point_estimate = model.estimate()
    assert model.point_estimate["2"].loc["treatment_time_2_1", "est"] == 0.0
    assert model.point_estimate["3"].loc["treatment_time_3_2", "est"] == 0.0


def test_db_event_study_uses_ref_minus_one_normalization(tmp_path):
    rows = []
    for unit, cohort in [(1, 2), (2, 3), (3, None), (4, 2), (5, None), (6, 3)]:
        unit_fe = 0.25 * unit
        for time in range(5):
            treated = int(cohort is not None and time >= cohort)
            rel_time = time - cohort if cohort is not None else None
            effect = (
                0.0
                if not treated
                else {0: 0.7, 1: 1.1, 2: 1.4}.get(rel_time, 1.4)
            )
            y = 2.0 + unit_fe + 0.4 * time + effect
            rows.append((unit, time, treated, y, unit))
    df = pd.DataFrame(rows, columns=["unit", "time", "D", "Y", "cluster"])
    con = _make_con(tmp_path, df)

    model = DBMundlakEventStudy(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_col="D",
        unit_col="unit",
        time_col="time",
        cluster_col="cluster",
        seed=44,
        n_bootstraps=0,
        pre_treat_interactions=True,
    )
    model.fit()
    estimates = model.summary()["point_estimate"]

    assert estimates["2"].index.tolist() == [
        "treatment_time_2_0",
        "treatment_time_2_1",
        "treatment_time_2_2",
        "treatment_time_2_3",
        "treatment_time_2_4",
    ]
    assert estimates["3"].index.tolist() == [
        "treatment_time_3_0",
        "treatment_time_3_1",
        "treatment_time_3_2",
        "treatment_time_3_3",
        "treatment_time_3_4",
    ]
    assert estimates["2"].loc["treatment_time_2_1", "est"] == 0.0
    assert estimates["3"].loc["treatment_time_3_2", "est"] == 0.0
    np.testing.assert_allclose(
        estimates["2"].loc[
            ["treatment_time_2_2", "treatment_time_2_3", "treatment_time_2_4"],
            "est",
        ],
        [0.7, 1.1, 1.4],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        estimates["3"].loc[["treatment_time_3_3", "treatment_time_3_4"], "est"],
        [0.7, 1.1],
        atol=1e-12,
    )
