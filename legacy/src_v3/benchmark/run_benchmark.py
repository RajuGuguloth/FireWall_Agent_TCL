import time
import torch
import numpy as np
import sys
from pathlib import Path
import json
import joblib

# Add src to path to import HybridFirewall
sys.path.append(str(Path(__file__).parent.parent))
from inference.hybrid_pipeline import HybridFirewall

def run_benchmark():
    print("=" * 60)
    print("ML Firewall Performance Benchmark")
    print("=" * 60)

    fw = HybridFirewall()
    
    # Load test data
    X_enh = np.load("data/splits/enhanced/X_test.npy")
    y_enh = np.load("data/splits/enhanced/y_test.npy")
    
    # Tier-1 Sample
    rf_sample = X_enh[0, :11]
    # Tier-2 Sample
    seq_sample = X_enh[:10]
    
    n_iterations = 500

    print(f"\nRunning {n_iterations} iterations for each tier...")

    # --- Benchmark Tier-1 (Random Forest) ---
    start_t1 = time.perf_counter()
    for _ in range(n_iterations):
        _ = fw.tier1_classify(rf_sample)
    end_t1 = time.perf_counter()
    
    t1_time = (end_t1 - start_t1) * 1000 / n_iterations
    t1_throughput = n_iterations / (end_t1 - start_t1)
    print(f"\n[Tier-1: Random Forest]")
    print(f"  Avg Latency  : {t1_time:.4f} ms")
    print(f"  Throughput   : {t1_throughput:.2f} req/s")

    # --- Benchmark Tier-2 (CNN+GRU) ---
    # Need to move to CPU for fair measurement if not already
    fw.cnn_gru.to("cpu")
    start_t2 = time.perf_counter()
    for _ in range(n_iterations):
        _ = fw.tier2_classify(seq_sample)
    end_t2 = time.perf_counter()
    
    t2_time = (end_t2 - start_t2) * 1000 / n_iterations
    t2_throughput = n_iterations / (end_t2 - start_t2)
    print(f"\n[Tier-2: CNN+GRU]")
    print(f"  Avg Latency  : {t2_time:.4f} ms")
    print(f"  Throughput   : {t2_throughput:.2f} req/s")

    # --- Benchmark Full Hybrid ---
    # Hybrid is T1, and only T2 if T1 doesn't meet confidence
    # For worst-case latency, we'll force T2
    start_hybrid = time.perf_counter()
    for _ in range(n_iterations):
        # use_tier2=True forces both tiers for measurement
        _ = fw.classify(rf_sample, seq_sample, use_tier2=True)
    end_hybrid = time.perf_counter()
    
    hybrid_time = (end_hybrid - start_hybrid) * 1000 / n_iterations
    hybrid_throughput = n_iterations / (end_hybrid - start_hybrid)
    print(f"\n[Hybrid Pipeline (Worst Case)]")
    print(f"  Avg Latency  : {hybrid_time:.4f} ms")
    print(f"  Throughput   : {hybrid_throughput:.2f} req/s")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
