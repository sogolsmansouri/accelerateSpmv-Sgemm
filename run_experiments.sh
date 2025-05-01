#!/usr/bin/env bash
# run_full_experiments_fixed.sh

echo "mode,impl,cpu_ms,gpu_ms,speedup,l2,cpu_mem,gpu_mem,matrix_size,matrix_A_size,matrix_B_size,nnz_A,nnz_B" \
  > benchmark_results_full.csv

densities=(0.0001 0.001 0.005 0.01)
spmv_sizes=(1000 2000 5000 10000 20000 50000)
spgemm_dims=(500 1000 2000 5000)

export OMP_NUM_THREADS=8

# 1) SpMV: CPU vs GPU
for d in "${densities[@]}"; do
  for s in "${spmv_sizes[@]}"; do
    echo "[SpMV] size=${s}, density=${d} — CPU"
    ./sp_ops -m spmv -i naive    -s $s -d $d >> benchmark_results_full.csv

    echo "[SpMV] size=${s}, density=${d} — GPU"
    ./sp_ops -m spmv -i gpu      -s $s -d $d >> benchmark_results_full.csv
  done
done

# 2) SpGEMM: only CPU serial & parallel
for d in "${densities[@]}"; do
  for N in "${spgemm_dims[@]}"; do
    echo "[SpGEMM] N=${N}, density=${d} — CPU serial"
    ./sp_ops -m spgemm -i naive    -a $N -b $N -d $d >> benchmark_results_full.csv

    echo "[SpGEMM] N=${N}, density=${d} — CPU parallel"
    ./sp_ops -m spgemm -i parallel -a $N -b $N -d $d >> benchmark_results_full.csv
  done
done

echo "Experiment complete. See benchmark_results_full.csv"
