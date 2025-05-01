#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read the benchmark results CSV file.
csv_filename = "benchmark_results.csv"
df = pd.read_csv(csv_filename)

# For clarity, separate data for SpMV and SpGEMM.
df_spmv = df[df['mode'] == 'spmv'].copy()
df_spgemm = df[df['mode'] == 'spgemm'].copy()


if 'matrix_size' in df_spmv.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spmv['matrix_size'], df_spmv['l2_norm_diff'], marker='o', color='purple')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('L2 Norm Difference')
    plt.title('SpMV: L2 Norm Difference vs. Matrix Size')
    plt.grid(True)
    plt.savefig("spmv_l2diff_vs_size.png")
    plt.show()
else:
    print("No 'matrix_size' field found for SpMV in CSV.")


if 'matrix_size' in df_spmv.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spmv['matrix_size'], df_spmv['cpu_mem_mb'], marker='o', color='blue')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('CPU Memory Usage (MB)')
    plt.title('SpMV: CPU Memory Usage vs. Matrix Size')
    plt.grid(True)
    plt.savefig("spmv_cpu_mem_vs_size.png")
    plt.show()
else:
    print("No 'matrix_size' field found for SpMV in CSV.")


if 'matrix_size' in df_spmv.columns and df_spmv['gpu_mem_mb'].max() > 0:
    plt.figure(figsize=(8,6))
    plt.plot(df_spmv['matrix_size'], df_spmv['gpu_mem_mb'], marker='o', color='red')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('SpMV: GPU Memory Usage vs. Matrix Size')
    plt.grid(True)
    plt.savefig("spmv_gpu_mem_vs_size.png")
    plt.show()
else:
    print("GPU memory usage is 0 or 'matrix_size' is not logged for SpMV.")


if 'matrix_A_size' in df_spgemm.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spgemm['matrix_A_size'], df_spgemm['l2_norm_diff'], marker='o', color='purple')
    plt.xlabel('Matrix A Size (rows)')
    plt.ylabel('L2 Norm Difference')
    plt.title('SpGEMM: L2 Norm Difference vs. Matrix A Size')
    plt.grid(True)
    plt.savefig("spgemm_l2diff_vs_size.png")
    plt.show()
else:
    print("No 'matrix_A_size' field found for SpGEMM in CSV.")


if 'matrix_A_size' in df_spgemm.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spgemm['matrix_A_size'], df_spgemm['cpu_mem_mb'], marker='o', color='blue')
    plt.xlabel('Matrix A Size (rows)')
    plt.ylabel('CPU Memory Usage (MB)')
    plt.title('SpGEMM: CPU Memory Usage vs. Matrix A Size')
    plt.grid(True)
    plt.savefig("spgemm_cpu_mem_vs_size.png")
    plt.show()
else:
    print("No 'matrix_A_size' field found for SpGEMM in CSV.")


if 'matrix_A_size' in df_spgemm.columns and df_spgemm['gpu_mem_mb'].max() > 0:
    plt.figure(figsize=(8,6))
    plt.plot(df_spgemm['matrix_A_size'], df_spgemm['gpu_mem_mb'], marker='o', color='red')
    plt.xlabel('Matrix A Size (rows)')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('SpGEMM: GPU Memory Usage vs. Matrix A Size')
    plt.grid(True)
    plt.savefig("spgemm_gpu_mem_vs_size.png")
    plt.show()
else:
    print("GPU memory usage is 0 or 'matrix_A_size' is not logged for SpGEMM.")


if 'matrix_size' in df_spmv.columns and 'speedup' in df_spmv.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spmv['matrix_size'], df_spmv['speedup'], marker='o', color='green')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title('SpMV: Speedup vs. Matrix Size')
    plt.grid(True)
    plt.savefig("spmv_speedup_vs_size.png")
    plt.show()
else:
    print("Speedup or 'matrix_size' data not found for SpMV.")


if 'matrix_A_size' in df_spgemm.columns and 'cpu_time_sec' in df_spgemm.columns and 'gpu_time_sec' in df_spgemm.columns:
    plt.figure(figsize=(8,6))
    plt.plot(df_spgemm['matrix_A_size'], df_spgemm['cpu_time_sec'], marker='o', color='blue', label='CPU Time')
    plt.plot(df_spgemm['matrix_A_size'], df_spgemm['gpu_time_sec'], marker='o', color='orange', label='GPU Time')
    plt.xlabel('Matrix A Size (rows)')
    plt.ylabel('Time (sec)')
    plt.title('SpGEMM: CPU vs. GPU Time vs. Matrix A Size')
    plt.legend()
    plt.grid(True)
    plt.savefig("spgemm_cpu_gpu_time_vs_size.png")
    plt.show()
else:
    print("Timing data not found for SpGEMM.")
