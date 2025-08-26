#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from duckreg.estimators import DuckRegression

def quick_benchmark():
    """Quick benchmark focusing on the compression step"""
    print("Running quick benchmark of optimized DuckRegression...")
    
    start_time = time.time()
    
    m = DuckRegression(
        db_name="large_dataset.db",
        table_name="data",
        formula="Y ~ D + f1 + f2", 
        cluster_col="",
        n_bootstraps=0,
        seed=42,
    )
    
    init_time = time.time()
    print(f"Initialization: {init_time - start_time:.4f}s")
    
    m.prepare_data()
    prepare_time = time.time()
    print(f"Data preparation: {prepare_time - init_time:.4f}s")
    
    m.compress_data()
    compress_time = time.time()
    print(f"Data compression: {compress_time - prepare_time:.4f}s")
    
    m.point_estimate = m.estimate()
    estimate_time = time.time()
    print(f"Estimation: {estimate_time - compress_time:.4f}s")
    
    total_time = estimate_time - start_time
    print(f"Total time: {total_time:.4f}s")
    
    # Print compression stats
    print(f"Compressed data shape: {m.df_compressed.shape}")
    print(f"Compressed query: {m.agg_query}")
    
    return m, total_time

if __name__ == "__main__":
    model, exec_time = quick_benchmark()