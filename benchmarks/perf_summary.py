#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from duckreg.estimators import DuckRegression

def profile_query_construction():
    """Profile just the query construction part of compress_data"""
    m = DuckRegression(
        db_name="large_dataset.db",
        table_name="data",
        formula="Y ~ D + f1 + f2",
        cluster_col="",
        n_bootstraps=0,
        seed=42,
    )
    m._parse_formula()

    # Time the optimized query construction
    start = time.perf_counter()
    for _ in range(1000):
        # Pre-compute expressions once to avoid repeated string operations
        group_by_cols = ", ".join(m.strata_cols)

        # Build aggregation expressions more efficiently
        agg_parts = ["COUNT(*) as count"]
        sum_expressions = []
        sum_sq_expressions = []

        for var in m.outcome_vars:
            sum_expr = f"SUM({var}) as sum_{var}"
            sum_sq_expr = f"SUM(POW({var}, 2)) as sum_{var}_sq"
            sum_expressions.append(sum_expr)
            sum_sq_expressions.append(sum_sq_expr)

        # Single join operation instead of multiple concatenations
        all_agg_expressions = ", ".join(agg_parts + sum_expressions + sum_sq_expressions)

        agg_query = f"""
        SELECT {group_by_cols}, {all_agg_expressions}
        FROM {m.table_name}
        GROUP BY {group_by_cols}
        """

    optimized_time = time.perf_counter() - start

    # Time the original approach
    start = time.perf_counter()
    for _ in range(1000):
        agg_expressions = ["COUNT(*) as count"]
        agg_expressions += [f"SUM({var}) as sum_{var}" for var in m.outcome_vars]
        agg_expressions += [
            f"SUM(POW({var}, 2)) as sum_{var}_sq" for var in m.outcome_vars
        ]
        group_by_cols = ", ".join(m.strata_cols)
        agg_query = f"""
        SELECT {group_by_cols}, {", ".join(agg_expressions)}
        FROM {m.table_name}
        GROUP BY {group_by_cols}
        """

    original_time = time.perf_counter() - start

    print("Query construction benchmark (1000 iterations):")
    print(f"  Original approach: {original_time:.4f}s")
    print(f"  Optimized approach: {optimized_time:.4f}s")
    print(f"  Speedup: {original_time/optimized_time:.2f}x")

def run_full_comparison():
    """Run full comparison showing the optimizations"""

    print("=== DuckRegression Performance Analysis ===")
    print()

    # Profile query construction
    profile_query_construction()
    print()

    # Run actual benchmark
    print("Full pipeline benchmark:")
    start_time = time.time()

    m = DuckRegression(
        db_name="large_dataset.db",
        table_name="data",
        formula="Y ~ D + f1 + f2",
        cluster_col="",
        n_bootstraps=0,
        seed=42,
    )
    m.fit()

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")
    return m, total_time

if __name__ == "__main__":
    model, exec_time = run_full_comparison()
