# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 1) Load the data
# df = pd.read_csv("benchmark_results_full.csv")

# # 2) Compute a density column for both workloads
# df['density'] = np.where(
#     df['mode']=='spmv',
#     df['nnz_A'] / (df['matrix_size']**2),
#     df['nnz_A'] / (df['matrix_A_size'] * df['matrix_B_size'])
# )

# # 3) Split into SpMV/SpGEMM and drop zeros for log plots
# df_spmv   = df[df['mode']=='spmv'].copy()
# df_spgemm = df[df['mode']=='spgemm'].copy()
# df_spmv[['cpu_ms','gpu_ms']]   = df_spmv[['cpu_ms','gpu_ms']].replace(0, np.nan)
# df_spgemm['cpu_ms']            = df_spgemm['cpu_ms'].replace(0, np.nan)

# # Helper: unique densities sorted
# dens_spmv   = sorted(df_spmv['density'].unique())
# dens_spgemm = sorted(df_spgemm['density'].unique())

# # ---- 1) SpMV: CPU vs GPU time ----
# fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# for ρ in dens_spmv:
#     cpu = df_spmv[(df_spmv['density']==ρ) & (df_spmv['impl']=='naive')]
#     gpu = df_spmv[(df_spmv['density']==ρ) & (df_spmv['impl']=='gpu' )]
#     if cpu['cpu_ms'].notna().any():
#         ax.plot(cpu['matrix_size'], cpu['cpu_ms'],   marker='o',
#                 label=f'CPU naïve, ρ={ρ:.4f}')
#     if gpu['gpu_ms'].notna().any():
#         ax.plot(gpu['matrix_size'], gpu['gpu_ms'],   marker='s', linestyle='--',
#                 label=f'GPU      , ρ={ρ:.4f}')
# ax.set(xscale='log', yscale='log',
#        xlabel='Matrix size (N)', ylabel='Time (ms)',
#        title='SpMV: CPU vs GPU Time')
# ax.grid(True, which='both', ls='--', alpha=0.5)
# ax.legend(loc='upper left', bbox_to_anchor=(1.02,1))
# fig.savefig('spmv_cpu_vs_gpu.png')

# # ---- 2) SpMV: Speedup (CPU/GPU) ----
# fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# for ρ in dens_spmv:
#     sub = df_spmv[df_spmv['density']==ρ]
#     ax.plot(sub['matrix_size'], sub['speedup'], marker='o', label=f'ρ={ρ:.4f}')
# ax.set(xscale='log',
#        xlabel='Matrix size (N)', ylabel='Speedup (CPU/GPU)',
#        title='SpMV: CPU/GPU Speedup')
# ax.grid(True, which='both', ls='--', alpha=0.5)
# ax.legend(loc='best')
# fig.savefig('spmv_speedup.png')

# # ---- 3) SpGEMM: CPU Serial vs Parallel ----
# fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# for ρ in dens_spgemm:
#     serial   = df_spgemm[(df_spgemm['density']==ρ) & (df_spgemm['impl']=='naive'   )]
#     parallel = df_spgemm[(df_spgemm['density']==ρ) & (df_spgemm['impl']=='parallel')]
#     if serial['cpu_ms'].notna().any():
#         ax.plot(serial['matrix_A_size'], serial['cpu_ms'], marker='o',
#                 label=f'Serial  , ρ={ρ:.4f}')
#     if parallel['cpu_ms'].notna().any():
#         ax.plot(parallel['matrix_A_size'], parallel['cpu_ms'], marker='s', linestyle='--',
#                 label=f'Parallel, ρ={ρ:.4f}')
# ax.set(xscale='log', yscale='log',
#        xlabel='Matrix dimension N', ylabel='Time (ms)',
#        title='SpGEMM: CPU Serial vs Parallel')
# ax.grid(True, which='both', ls='--', alpha=0.5)
# ax.legend(loc='upper left', bbox_to_anchor=(1.02,1))
# fig.savefig('spgemm_cpu_serial_parallel.png')

# print("Plots saved:")
# print(" • spmv_cpu_vs_gpu.png")
# print(" • spmv_speedup.png")
# print(" • spgemm_cpu_serial_parallel.png")
# import matplotlib.pyplot as plt
# import pandas as pd

# # Data for overhead breakdown
# overhead_data = pd.DataFrame({
#     'CPU_prep': [0.05, 0.01],  # fractions
#     'Transfer': [0.60, 0.04],
#     'Kernel': [0.35, 0.95]
# }, index=['N=1000', 'N=10000'])

# # Plot stacked bar for overhead
# plt.figure()
# overhead_data.plot(kind='bar', stacked=True, legend=True)
# plt.ylabel('Fraction of GPU Time')
# plt.title('SpMV GPU Overhead Breakdown')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.savefig('overhead_breakdown.pdf')

# # Data for CPU scaling
# cores = [1,2,3,4,5,6,7,8]
# speedup = [1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.4]

# # Plot speedup vs cores
# plt.figure()
# plt.plot(cores, speedup, marker='o')
# plt.xlabel('Number of CPU Cores')
# plt.ylabel('Speedup (vs 1 core)')
# plt.title('Tiled SpGEMM Multi-Core Scaling')
# plt.xticks(cores)
# plt.tight_layout()
# plt.savefig('cpu_scaling.pdf')
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# ---- Configuration ----
CSV_PATH = "benchmark_results.csv"
OUTPUT_DIR = "."  # where to write the PDFs

# ---- Load data ----
df = pd.read_csv(CSV_PATH)

# ---- SpMV plot ----
spmv = df[df['mode']=="spmv"]
# ensure ordered by matrix_size
spmv = spmv.sort_values("matrix_size")

plt.figure(figsize=(6,4))
plt.plot(spmv['matrix_size'], spmv['cpu_time_sec'], marker='o', label='CPU')
plt.plot(spmv['matrix_size'], spmv['gpu_time_sec'], marker='o', label='GPU')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (s)')
plt.title('SpMV: CPU vs GPU Time')
plt.legend()
plt.grid(which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spmv_cpu_gpu_times.pdf")
plt.close()

# ---- SpGEMM plot ----
spgemm = df[df['mode']=="spgemm"]
# order by matrix_A_size
spgemm = spgemm.sort_values("matrix_A_size")

plt.figure(figsize=(6,4))
plt.plot(spgemm['matrix_A_size'], spgemm['cpu_time_sec'], marker='o', label='CPU')
plt.plot(spgemm['matrix_A_size'], spgemm['gpu_time_sec'], marker='o', label='GPU')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix A Size (rows)')
plt.ylabel('Time (s)')
plt.title('SpGEMM: CPU vs GPU Time')
plt.legend()
plt.grid(which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spgemm_cpu_gpu_times.pdf")
plt.close()

print("Plots written to:")
print(f"  - {OUTPUT_DIR}/spmv_cpu_gpu_times.pdf")
print(f"  - {OUTPUT_DIR}/spgemm_cpu_gpu_times.pdf")
