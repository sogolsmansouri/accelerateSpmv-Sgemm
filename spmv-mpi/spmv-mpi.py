#!/usr/bin/env python3
"""
Accelerating Fundamental Operations Using Parallel Computing
=============================================================

This script demonstrates accelerated sparse operations:

  * SpMV  - Sparse Matrix-Vector Multiplication
  * SpGEMM - Sparse Matrix-Matrix Multiplication

It provides three implementations:
  1. A naive CPU implementation using SciPy.
  2. A tiled, parallel CPU approach (for SpGEMM) using Python's multiprocessing.
  3. A GPU-accelerated implementation using CuPy.

Additional benchmarking metrics include:
  - Execution Time (CPU & GPU)
  - Maximum CPU Memory Usage (via resource.getrusage)
  - GPU Memory Usage (using CuPy's memory pool)
  - Matrix sizes and nonzero counts for further insight

Benchmark results are printed to the console and saved in "benchmark_results.csv".

Usage examples:
  - SpMV (using a matrix file or generating a random matrix and vector):
      python sp_ops_csv.py --mode spmv --matrix_path data/your_matrix.mtx
      python sp_ops_csv.py --mode spmv --size 10000 --density 0.001 --impl gpu

  - SpGEMM (generating two random matrices):
      python sp_ops_csv.py --mode spgemm --sizeA 2000 --sizeB 2000 --density 0.01 --impl gpu
      python sp_ops_csv.py --mode spgemm --matrix_path data/A.mtx data/B.mtx

The script prints execution times, computes speedups (GPU vs CPU), reports the L2 norm difference 
between CPU and GPU results, and saves all metrics to benchmark_results.csv.
"""

import argparse
import time
import numpy as np
import sys
from scipy.io import mmread
from scipy.sparse import csr_matrix, random as sparse_random
import resource
import csv
from mpi4py import MPI

# Try importing CuPy; if not available, GPU routines will fail.
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None

###############################
# CSV Logging Helper Function
###############################
def log_results_csv(results, filename="benchmark_results.csv"):
    """Append benchmark results (list of dictionaries) to a CSV file."""
    fieldnames = [
        "mode", "impl", "cpu_time_sec", "gpu_time_sec", "speedup", "l2_norm_diff",
        "cpu_mem_mb", "gpu_mem_mb", "matrix_size", "matrix_A_size", "matrix_B_size",
        "nnz_A", "nnz_B"
    ]
    file_exists = False
    try:
        with open(filename, "r") as csvfile:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

###############################
# CPU Implementations
###############################
def spmv_cpu(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    """Naive CPU-based SpMV: y = A*x"""
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data
    m = A.shape[0]
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        for idx in range(A_indptr[i], A_indptr[i + 1]):
            y[i] += A_data[idx] * x[A_indices[idx]]
    return y

def spmv_cpu_mpi(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    """ CPU SpMV version using MPI function call for parallelization """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    m = A.shape[0]
    
    # Calculate row distribution
    rows_per_proc = m // size
    remainder = m % size
    
    # Calculate start and end rows for this process
    start_row = rank * rows_per_proc + min(rank, remainder)
    end_row = start_row + rows_per_proc + (1 if rank < remainder else 0)
    local_rows = end_row - start_row
    
    A = comm.bcast(A, root=0)
    x = comm.bcast(x, root=0)
    
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data
    
    local_y = np.zeros(local_rows, dtype=np.float32)
    
    for i in range(local_rows):
        global_i = i + start_row
        for idx in range(A_indptr[global_i], A_indptr[global_i + 1]):
            local_y[i] += A_data[idx] * x[A_indices[idx]]
    
    recv_counts = comm.allgather(local_rows)
    
    displacements = [0]
    for count in recv_counts[:-1]:
        displacements.append(displacements[-1] + count)
    
    y = np.zeros(m, dtype=np.float32)
    
    comm.Gatherv(sendbuf=local_y, recvbuf=(y, recv_counts, displacements, MPI.FLOAT), root=0)
    
    return y


def spgemm_cpu(A: csr_matrix, B: csr_matrix):
    """Naive CPU-based SpGEMM: C = A*B using SciPy's dot()."""
    start = time.time()
    C = A.dot(B)
    cpu_time = time.time() - start
    return C, cpu_time

def spgemm_cpu_tiled(A: csr_matrix, B: csr_matrix, n_workers: int = 4, tile_size: int = 0):
    """
    Tiled, parallel CPU SpGEMM.
    Splits matrix A by rows and multiplies each tile by B in parallel.
    Returns (C, duration) where C is a CSR matrix.
    """
    import multiprocessing
    nrows_A, _ = A.shape
    _, ncols_B = B.shape
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    shapeA = A.shape
    shapeB = B.shape

    chunk_size = (nrows_A + n_workers - 1) // n_workers if tile_size <= 0 else tile_size
    tasks = []
    row_start = 0
    while row_start < nrows_A:
        row_end = min(row_start + chunk_size, nrows_A)
        tasks.append((row_start, row_end, A_data, A_indices, A_indptr, 
                      B.data, B.indices, B.indptr, shapeA, shapeB))
        row_start = row_end

    def worker_tile(args):
        (r_start, r_end, A_data, A_indices, A_indptr,
         B_data, B_indices, B_indptr, shapeA, shapeB) = args
        from scipy.sparse import csr_matrix
        partial = {}
        for i in range(r_start, r_end):
            row_result = {}
            for idx in range(A_indptr[i], A_indptr[i + 1]):
                a_col = A_indices[idx]
                a_val = A_data[idx]
                for j in range(B_indptr[a_col], B_indptr[a_col + 1]):
                    b_col = B_indices[j]
                    b_val = B_data[j]
                    row_result[b_col] = row_result.get(b_col, 0) + a_val * b_val
            partial[i] = row_result
        return partial

    start = time.time()
    pool = multiprocessing.Pool(processes=n_workers)
    partial_results = pool.map(worker_tile, tasks)
    pool.close()
    pool.join()

    combined = {}
    for part in partial_results:
        combined.update(part)

    data, indices, indptr = [], [], [0]
    for i in range(nrows_A):
        row_dict = combined.get(i, {})
        for col in sorted(row_dict.keys()):
            data.append(row_dict[col])
            indices.append(col)
        indptr.append(len(data))
    C = csr_matrix((np.array(data, dtype=np.float32),
                    np.array(indices, dtype=np.int32),
                    np.array(indptr, dtype=np.int32)),
                   shape=(nrows_A, ncols_B))
    cpu_time = time.time() - start
    return C, cpu_time

###############################
# GPU Implementations (using CuPy)
###############################
def spmv_gpu(A_csr: csr_matrix, x: np.ndarray):
    """
    GPU-based SpMV using CuPy.
    Converts SciPy CSR matrix to CuPy CSR matrix and multiplies by vector.
    """
    if cp is None:
        raise ImportError("CuPy is not available for GPU implementation.")
    A_gpu = cpx_sparse.csr_matrix((cp.asarray(A_csr.data),
                                   cp.asarray(A_csr.indices),
                                   cp.asarray(A_csr.indptr)),
                                  shape=A_csr.shape)
    x_gpu = cp.asarray(x)
    _ = A_gpu.dot(x_gpu).get()  # Warm-up
    start = time.time()
    y_gpu = A_gpu.dot(x_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    return y_gpu.get(), gpu_time
    
def spmv_mpi_gpu(A_csr: csr_matrix, x: np.ndarray):
    """
    Hybrid MPI+CUDA implementation of Sparse Matrix-Vector multiplication.
    
    Each MPI process handles a portion of the matrix using its GPU.
    The matrix rows are distributed across processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if cp is None:
        raise ImportError("CuPy is not available for GPU implementation.")

    if rank == 0:
        matrix_data = {
            'data': A_csr.data,
            'indices': A_csr.indices,
            'indptr': A_csr.indptr,
            'shape': A_csr.shape
        }
    else:
        matrix_data = None
    
    # Broadcast matrix data and vector x
    matrix_data = comm.bcast(matrix_data, root=0)
    x = comm.bcast(x, root=0)

    if rank != 0:
        A_csr = csr_matrix((matrix_data['data'], matrix_data['indices'], matrix_data['indptr']),
                       shape=matrix_data['shape'])

    n_rows = A_csr.shape[0]
    rows_per_process = n_rows // size
    start_row = rank * rows_per_process
    end_row = ((rank + 1) * rows_per_process) if rank < size - 1 else n_rows

    # Extract local portion of the matrix
    if start_row < end_row:  # Ensure process has work to do
        local_A = A_csr[start_row:end_row, :]
    else:
        # Empty matrix for processes that don't get any rows
        local_A = csr_matrix((0, A_csr.shape[1]))

    x = comm.bcast(x, root=0)

    # Convert local data to GPU
    local_A_gpu = cpx_sparse.csr_matrix((cp.asarray(local_A.data),
                                         cp.asarray(local_A.indices),
                                         cp.asarray(local_A.indptr)),
                                        shape=local_A.shape)
    x_gpu = cp.asarray(x)

    # Warm-up run
    _ = local_A_gpu.dot(x_gpu).get()

    comm.Barrier()
    start = time.time()

    # Perform local SpMV on each GPU
    local_result_gpu = local_A_gpu.dot(x_gpu)
    local_result = local_result_gpu.get()

    cp.cuda.Stream.null.synchronize()

    if rank == 0:
        # Prepare buffer for gathering results
        result_counts = [rows_per_process] * (size - 1) + [n_rows - (size - 1) * rows_per_process]
        result_displacements = [i * rows_per_process for i in range(size)]
        result = np.empty(n_rows, dtype=x.dtype)
    else:
        result_counts = None
        result_displacements = None
        result = None

    comm.Gatherv(sendbuf=local_result, recvbuf=(result, result_counts, result_displacements, MPI.DOUBLE) if rank == 0 else None, root=0)

    comm.Barrier()
    end_time = time.time() - start

    max_time = comm.allreduce(end_time, op=MPI.MAX)

    return result, max_time

def spgemm_gpu(A_csr: csr_matrix, B_csr: csr_matrix):
    """
    GPU-based SpGEMM using CuPy.
    Converts two SciPy CSR matrices to CuPy and multiplies them.
    """
    if cp is None:
        raise ImportError("CuPy is not available for GPU implementation.")
    A_gpu = cpx_sparse.csr_matrix((cp.asarray(A_csr.data),
                                   cp.asarray(A_csr.indices),
                                   cp.asarray(A_csr.indptr)),
                                  shape=A_csr.shape)
    B_gpu = cpx_sparse.csr_matrix((cp.asarray(B_csr.data),
                                   cp.asarray(B_csr.indices),
                                   cp.asarray(B_csr.indptr)),
                                  shape=B_csr.shape)
    _ = A_gpu.dot(B_gpu).get()  # Warm-up
    start = time.time()
    C_gpu = A_gpu.dot(B_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    C = csr_matrix((C_gpu.data.get(),
                    C_gpu.indices.get(),
                    C_gpu.indptr.get()), shape=C_gpu.shape)
    return C, gpu_time

###############################
# Evaluation Helpers
###############################
def get_cpu_memory_usage_mb():
    """Return maximum resident set size (max RSS) in MB."""
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024

def get_gpu_memory_usage_mb():
    """Return GPU memory usage (if available) in MB."""
    if cp is None:
        return None
    try:
        used_bytes = cp.get_default_memory_pool().used_bytes()
        return used_bytes / (1024 ** 2)
    except Exception:
        return None

###############################
# Main Function
###############################
def main():
    parser = argparse.ArgumentParser(description="Accelerated Sparse Operations using Parallel Computing")
    parser.add_argument("--mode", type=str, default="spmv",
                        choices=["spmv", "spgemm"],
                        help="Operation mode: 'spmv' for sparse matrix-vector, 'spgemm' for sparse matrix-matrix multiplication.")
    parser.add_argument("--matrix_path", nargs="+", type=str,
                        help=("Path(s) to .mtx file(s): for spmv, provide one file; for spgemm, provide two files (A and B). If not provided, random matrices/vectors will be generated."))
    parser.add_argument("--size", type=int, default=1000,
                        help="Matrix size for random generation (square matrix for spmv).")
    parser.add_argument("--sizeA", type=int, default=1000,
                        help="Number of rows for matrix A in SpGEMM (if generating random matrices).")
    parser.add_argument("--sizeB", type=int, default=1000,
                        help="Number of columns for matrix A / rows for matrix B in SpGEMM (if generating random matrices).")
    parser.add_argument("--density", type=float, default=0.01,
                        help="Density for random sparse matrices.")
    parser.add_argument("--cpus", type=int, default=4,
                        help="Number of CPU processes for tiled SpGEMM.")
    parser.add_argument("--tile_size", type=int, default=0,
                        help="Tile size for parallel CPU SpGEMM (0 to auto-chunk).")
    parser.add_argument("--impl", type=str, default="gpu",
                        choices=["naive", "parallel", "gpu", "mpi", "mpi-gpu"],
                        help=("Implementation type for the selected mode: 'naive' for basic CPU, 'parallel' for tiled CPU (SpGEMM only), 'gpu' for CuPy acceleration."))
    args = parser.parse_args()

    cpu_mem = get_cpu_memory_usage_mb()
    gpu_mem = get_gpu_memory_usage_mb()

    # 1. Data Preparation
    if args.mode == "spmv":
        if args.matrix_path and len(args.matrix_path) >= 1:
            print(f"Reading matrix from {args.matrix_path[0]} ...")
            A = mmread(args.matrix_path[0]).tocsr().astype(np.float32)
        else:
            print(f"Generating random {args.size} x {args.size} sparse matrix (density = {args.density}) ...")
            rng = np.random.default_rng(seed=123)
            A = sparse_random(args.size, args.size, density=args.density, format='csr', random_state=rng, dtype=np.float32)
        nrows, ncols = A.shape
        matrix_size = nrows  # Since it's square.
        print(f"Matrix shape: {nrows} x {ncols}, nnz: {A.nnz}")
        x = np.random.rand(ncols).astype(np.float32)
    elif args.mode == "spgemm":
        if args.matrix_path and len(args.matrix_path) >= 2:
            print(f"Reading matrix A from {args.matrix_path[0]} ...")
            A = mmread(args.matrix_path[0]).tocsr().astype(np.float32)
            print(f"Reading matrix B from {args.matrix_path[1]} ...")
            B = mmread(args.matrix_path[1]).tocsr().astype(np.float32)
        else:
            print(f"Generating random matrix A ({args.sizeA} x {args.sizeB}) and matrix B ({args.sizeB} x {args.sizeA}), density = {args.density} ...")
            rng = np.random.default_rng(seed=123)
            A = sparse_random(args.sizeA, args.sizeB, density=args.density, format='csr', random_state=rng, dtype=np.float32)
            B = sparse_random(args.sizeB, args.sizeA, density=args.density, format='csr', random_state=rng, dtype=np.float32)
        matrix_A_size = A.shape[0]
        matrix_B_size = B.shape[1]  # For multiplication, assuming A is (m,n) and B is (n,p)
        print(f"Matrix A shape: {A.shape}, nnz: {A.nnz}")
        print(f"Matrix B shape: {B.shape}, nnz: {B.nnz}")

    # 2. Compute and Benchmark
    print("\n--- Running Operation ---")
    if args.mode == "spmv":
        if args.impl == "naive":
            print("\nRunning CPU-based (naive) SpMV ...")
            start_cpu = time.time()
            y_cpu = spmv_cpu(A, x)
            cpu_time = time.time() - start_cpu
            result_cpu = y_cpu
            print(f"CPU SpMV time: {cpu_time:.6f} seconds")
            gpu_time = 0.0
            diff = 0.0
            speedup = 1.0
        elif args.impl == "mpi":
            print("\nRunning CPU-based (MPI) SpMV ...")
            start_cpu = time.time()
            y_cpu = spmv_cpu_mpi(A, x)
            cpu_time = time.time() - start_cpu
            result_cpu = y_cpu
            print(f"CPU MPI SpMV time: {cpu_time:.6f} seconds")
            gpu_time = 0.0
            diff = 0.0
            speedup = 1.0
        elif args.impl == "gpu":
            if cp is None:
                sys.exit("Error: CuPy is not available for GPU implementation.")
            print("\nRunning GPU-based SpMV (CuPy) ...")
            y_gpu, gpu_time = spmv_gpu(A, x)
            cp.cuda.Stream.null.synchronize()
            print(f"GPU SpMV time: {gpu_time:.6f} seconds")
            result_cpu = spmv_cpu(A, x)
            start_cpu = time.time()
            _ = spmv_cpu(A, x)
            cpu_time = time.time() - start_cpu
            diff = np.linalg.norm(result_cpu - y_gpu)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"SpMV L2 norm difference: {diff:e}")
        elif args.impl == "mpi-gpu":
            if cp is None:
                sys.exit("Error: CuPy is not available for GPU implementation.")
            print("\nRunning GPU-based SpMV (CuPy) ...")
            y_gpu, gpu_time = spmv_mpi_gpu(A, x)
            cp.cuda.Stream.null.synchronize()
            print(f"GPU SpMV time: {gpu_time:.6f} seconds")
            result_cpu = spmv_cpu(A, x)
            start_cpu = time.time()
            _ = spmv_cpu(A, x)
            cpu_time = time.time() - start_cpu
            diff = np.linalg.norm(result_cpu - y_gpu)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"SpMV L2 norm difference: {diff:e}")
        else:
            sys.exit("Invalid implementation selected for SpMV. Choose 'naive' or 'gpu'.")
    elif args.mode == "spgemm":
        if args.impl == "naive":
            print("\nRunning CPU-based (naive) SpGEMM ...")
            C_cpu, cpu_time = spgemm_cpu(A, B)
            result_cpu = C_cpu
            gpu_time = 0.0
            diff = 0.0
            speedup = 1.0
            print(f"Naive CPU SpGEMM time: {cpu_time:.6f} seconds")
        elif args.impl == "parallel":
            print(f"\nRunning tiled, parallel CPU SpGEMM with {args.cpus} processes ...")
            C_cpu, cpu_time = spgemm_cpu_tiled(A, B, n_workers=args.cpus, tile_size=args.tile_size)
            result_cpu = C_cpu
            gpu_time = 0.0
            diff = 0.0
            speedup = 1.0
            print(f"Parallel CPU SpGEMM time: {cpu_time:.6f} seconds")
        elif args.impl == "gpu":
            if cp is None:
                sys.exit("Error: CuPy is not available for GPU implementation.")
            print("\nRunning GPU-based SpGEMM (CuPy) ...")
            C_gpu, gpu_time = spgemm_gpu(A, B)
            result_cpu, cpu_time = spgemm_cpu(A, B)
            diff = np.linalg.norm(result_cpu.toarray() - C_gpu.toarray())
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"GPU SpGEMM time: {gpu_time:.6f} sec")
            print(f"Speedup: {speedup:.2f}x")
            print(f"SpGEMM L2 norm difference: {diff:e}")
        else:
            sys.exit("Invalid implementation selected for SpGEMM. Choose 'naive', 'parallel', or 'gpu'.")

    # 3. Benchmark Summary
    print("\n--- Benchmark Summary ---")
    if args.mode == "spmv":
        if args.impl == "gpu":
            print(f"CPU SpMV time: {cpu_time:.6f} sec")
            print(f"GPU SpMV time: {gpu_time:.6f} sec")
            print(f"Speedup: {speedup:.2f}x")
        else:
            print(f"CPU SpMV time (naive): {cpu_time:.6f} sec")
        # For CSV logging, matrix_A_size and matrix_B_size are not applicable for SpMV.
        results = [{
            "mode": args.mode,
            "impl": args.impl,
            "cpu_time_sec": cpu_time,
            "gpu_time_sec": gpu_time,
            "speedup": speedup,
            "l2_norm_diff": diff,
            "cpu_mem_mb": get_cpu_memory_usage_mb(),
            "gpu_mem_mb": get_gpu_memory_usage_mb() if cp is not None else 0,
            "matrix_size": matrix_size,
            "matrix_A_size": 0,
            "matrix_B_size": 0,
            "nnz_A": A.nnz,
            "nnz_B": 0
        }]
    elif args.mode == "spgemm":
        if args.impl == "gpu":
            print(f"CPU SpGEMM time (naive): {cpu_time:.6f} sec")
            print(f"GPU SpGEMM time: {gpu_time:.6f} sec")
            print(f"Speedup: {speedup:.2f}x")
        else:
            print(f"CPU SpGEMM time: {cpu_time:.6f} sec")
        results = [{
            "mode": args.mode,
            "impl": args.impl,
            "cpu_time_sec": cpu_time,
            "gpu_time_sec": gpu_time,
            "speedup": speedup,
            "l2_norm_diff": diff,
            "cpu_mem_mb": get_cpu_memory_usage_mb(),
            "gpu_mem_mb": get_gpu_memory_usage_mb() if cp is not None else 0,
            "matrix_size": 0,
            "matrix_A_size": A.shape[0],
            "matrix_B_size": B.shape[1],
            "nnz_A": A.nnz,
            "nnz_B": B.nnz
        }]
    
    log_results_csv(results)
    print("\nBenchmark results saved to 'benchmark_results.csv'.")

if __name__ == "__main__":
    main()
