import pytest
import numpy as np
from duckreg.estimators import DuckRegression
from tests.utils import generate_sample_data, create_duckdb_database
import pyfixest as pf

@pytest.fixture(scope="session")
def get_data():
    return generate_sample_data()

@pytest.fixture(scope="session")
def database(get_data):
    df = get_data
    db_name = 'test_dataset.db'
    create_duckdb_database(df, db_name)
    return df


@pytest.mark.parametrize("fml", ["Y ~ D", "Y ~ D + f1", "Y ~ D + f1 + f2"])
@pytest.mark.parametrize("cluster_col", [""])

def test_vs_pyfixest_deterministic(get_data, fml, cluster_col):

    m_duck = DuckRegression(
        db_name='test_dataset.db',
        table_name='data',
        formula=fml,
        cluster_col=cluster_col,
        n_bootstraps=0,
        seed = 42
    )
    m_duck.fit()
    m_duck.fit_vcov()

    m_feols = pf.feols(
        fml,
        data = get_data,
        vcov = "hetero" if cluster_col == "" else {"CRV1": cluster_col},
        ssc = pf.ssc(adj = False, cluster_adj = True)
    )

    results = m_duck.summary()
    coefs = results["point_estimate"]
    se = results["standard_error"]

    assert np.all(np.abs(coefs) - np.abs(m_feols.coef().values) < 1e-8), "Coeficients are not equal"
    assert np.all(np.abs(se) - np.abs(m_feols.se().values) < 1e-4), "Standard errors are not equal"

def test_multiple_estimation_stochastic():

    pass


def test_vs_pyfixest_stochastic():

    pass


def test_mundlak():

    pass


def test_double_demeaning():

    pass