"""
Coverage simulation for DuckDML cluster bootstrap.

Tests whether the 95% confidence intervals have the correct coverage rate
under both homoskedastic and heteroskedastic error structures.
"""
import numpy as np
import pandas as pd
import duckdb
from duckreg.estimators import DuckDML
from tqdm import tqdm
import os

def generate_data(n, n_towns, n_days, true_beta, hetero_type='none', seed=None):
    """
    Generate data for the partially linear model.

    Parameters:
    -----------
    hetero_type : str
        'none': homoskedastic errors
        'treatment': heteroskedasticity proportional to |X|
        'group': heteroskedasticity varies by group
        'both': both treatment and group heteroskedasticity
    """
    if seed is not None:
        np.random.seed(seed)

    df = pd.DataFrame({
        'town_id': np.random.randint(0, n_towns, n),
        'day_id': np.random.randint(0, n_days, n),
    })

    # Nonlinear function of covariates
    g_z = 0.5 * df['town_id'] + 0.01 * df['day_id'] * df['town_id'] + np.sin(df['day_id'])

    # Treatment correlated with fixed effects
    df['X'] = 0.2 * df['town_id'] + 0.1 * df['day_id'] + np.random.randn(n)

    # Generate heteroskedastic errors
    if hetero_type == 'none':
        # Homoskedastic: constant variance
        errors = np.random.normal(0, 2, n)
    elif hetero_type == 'treatment':
        # Variance proportional to |X|
        sigma_i = 1 + 0.5 * np.abs(df['X'])
        errors = np.random.normal(0, 1, n) * sigma_i
    elif hetero_type == 'group':
        # Variance varies by town (group-level heteroskedasticity)
        sigma_g = 1 + 0.3 * (df['town_id'] % 10)
        errors = np.random.normal(0, 1, n) * sigma_g
    elif hetero_type == 'both':
        # Both treatment and group heteroskedasticity
        sigma_i = (1 + 0.3 * (df['town_id'] % 10)) * (1 + 0.2 * np.abs(df['X']))
        errors = np.random.normal(0, 1, n) * sigma_i
    else:
        raise ValueError(f"Unknown hetero_type: {hetero_type}")

    # Outcome follows partially linear model
    df['Y'] = true_beta * df['X'] + g_z + errors

    return df

def run_simulation(n_sims, n, n_towns, n_days, true_beta, hetero_type,
                   n_bootstraps=200, seed_start=0):
    """
    Run coverage simulation.
    """
    coverage_count = 0
    estimates = []
    std_errors = []
    ci_widths = []

    for sim in tqdm(range(n_sims), desc=f"Coverage sim ({hetero_type})"):
        # Generate data
        df = generate_data(n, n_towns, n_days, true_beta, hetero_type,
                          seed=seed_start + sim)

        # Create temporary database
        db_name = f"temp_sim_{sim}.db"
        table_name = "data"

        try:
            con = duckdb.connect(db_name)
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.register('df_pandas', df)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_pandas")
            con.close()

            # Run DML
            dml = DuckDML(
                db_name=db_name,
                table_name=table_name,
                outcome_var='Y',
                treatment_var='X',
                discrete_covars=['town_id', 'day_id'],
                seed=42,
                n_bootstraps=n_bootstraps
            )

            dml.fit()
            results = dml.summary()

            beta_hat = results['point_estimate'][0]
            se = results['standard_error'].flatten()[0]

            # Construct 95% CI using normal approximation
            ci_lower = beta_hat - 1.96 * se
            ci_upper = beta_hat + 1.96 * se

            # Check coverage
            if ci_lower <= true_beta <= ci_upper:
                coverage_count += 1

            estimates.append(beta_hat)
            std_errors.append(se)
            ci_widths.append(ci_upper - ci_lower)

        finally:
            # Clean up
            if os.path.exists(db_name):
                os.remove(db_name)

    coverage_rate = coverage_count / n_sims

    return {
        'coverage_rate': coverage_rate,
        'mean_estimate': np.mean(estimates),
        'std_estimate': np.std(estimates),
        'mean_se': np.mean(std_errors),
        'mean_ci_width': np.mean(ci_widths),
        'estimates': estimates,
        'std_errors': std_errors
    }

# Simulation parameters
N_SIMS = 100
N = 50_000
N_TOWNS = 100
N_DAYS = 50
TRUE_BETA = 2.5
N_BOOTSTRAPS = 200

print("="*70)
print("DuckDML Coverage Simulation")
print("="*70)
print(f"Number of simulations: {N_SIMS}")
print(f"Sample size per simulation: {N:,}")
print(f"True beta: {TRUE_BETA}")
print(f"Bootstrap replications: {N_BOOTSTRAPS}")
print(f"Nominal coverage: 95%")
print("="*70)

# Test different error structures
error_structures = {
    'Homoskedastic': 'none',
    'Treatment heteroskedasticity': 'treatment',
    'Group heteroskedasticity': 'group',
    'Both': 'both'
}

results_all = {}

for name, hetero_type in error_structures.items():
    print(f"\n{name}")
    print("-" * 70)

    results = run_simulation(
        n_sims=N_SIMS,
        n=N,
        n_towns=N_TOWNS,
        n_days=N_DAYS,
        true_beta=TRUE_BETA,
        hetero_type=hetero_type,
        n_bootstraps=N_BOOTSTRAPS,
        seed_start=1000 * list(error_structures.keys()).index(name)
    )

    results_all[name] = results

    print(f"Coverage rate:        {results['coverage_rate']:.1%}")
    print(f"Mean estimate:        {results['mean_estimate']:.4f} (bias: {results['mean_estimate'] - TRUE_BETA:.4f})")
    print(f"Std dev of estimates: {results['std_estimate']:.4f}")
    print(f"Mean SE (bootstrap):  {results['mean_se']:.4f}")
    print(f"SE ratio:             {results['mean_se'] / results['std_estimate']:.3f}")
    print(f"Mean CI width:        {results['mean_ci_width']:.4f}")

# Summary table
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Error Structure':<30} {'Coverage':>12} {'SE Ratio':>12} {'Bias':>12}")
print("-" * 70)

for name in error_structures.keys():
    r = results_all[name]
    se_ratio = r['mean_se'] / r['std_estimate']
    bias = r['mean_estimate'] - TRUE_BETA
    print(f"{name:<30} {r['coverage_rate']:>11.1%} {se_ratio:>12.3f} {bias:>12.4f}")

print("="*70)
print("\nInterpretation:")
print("- Coverage should be close to 95% for valid inference")
print("- SE ratio should be close to 1.0 (bootstrap SE matches empirical SD)")
print("- Bias should be close to 0.0")
print("\nNote: Cluster bootstrap should provide valid inference even under")
print("heteroskedasticity, unlike the homoskedastic theory in DM95 Theorem 2.")
