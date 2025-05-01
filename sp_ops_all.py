#!/usr/bin/env python3
"""
Accelerating Fundamental Operations Using Parallel Computing (with Numba)
=======================================================================

- SpMV via:
    1. naive CPU
    2. Numba GPU kernel
- SpGEMM via:
    1. naive CPU
    2. (optional) you can add a Numba kernel later

Results saved to benchmark_results.csv
"""

import argparse, time, csv, sys
import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random
from scipy.io import mmread
import resource

# try to import Numba
try:
    from numba import cuda, float32, int32
except ImportError:
    cuda = None

##############################
# CSV Logging
##############################
def log_results_csv(results, filename="benchmark_results.csv"):
    fieldnames = [
        "mode","impl","cpu_time_sec","gpu_time_sec","speedup","l2_norm_diff",
        "cpu_mem_mb","gpu_mem_mb","matrix_size","matrix_A_size","matrix_B_size",
        "nnz_A","nnz_B"
    ]
    exists = False
    try:
        with open(filename,"r"): exists=True
    except FileNotFoundError:
        pass
    with open(filename,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        if not exists: w.writeheader()
        for r in results: w.writerow(r)

##############################
# CPU SpMV
##############################
def spmv_cpu(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    indptr, indices, data = A.indptr, A.indices, A.data
    m = A.shape[0]
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        s = 0.0
        for idx in range(indptr[i], indptr[i+1]):
            s += data[idx] * x[indices[idx]]
        y[i] = s
    return y

##############################
# Numba CUDA SpMV kernel
##############################
if cuda:
    @cuda.jit
    def _spmv_kernel(indptr, indices, data, x, y):
        row = cuda.grid(1)
        if row < indptr.shape[0]-1:
            s = 0.0
            start = indptr[row]
            end   = indptr[row+1]
            for j in range(start, end):
                s += data[j] * x[ indices[j] ]
            y[row] = s


def spmv_numba(A: csr_matrix, x: np.ndarray):
    if cuda is None:
        raise ImportError("Numba/CUDA not available")
    # transfer
    indptr_d = cuda.to_device(A.indptr.astype(np.int32))
    indices_d= cuda.to_device(A.indices.astype(np.int32))
    data_d   = cuda.to_device(A.data.astype(np.float32))
    x_d      = cuda.to_device(x.astype(np.float32))
    y_d      = cuda.device_array(A.shape[0], dtype=np.float32)

    threads = 128
    blocks  = (A.shape[0] + threads - 1)//threads

    # warm‐up
    _spmv_kernel[blocks,threads](indptr_d, indices_d, data_d, x_d, y_d)
    cuda.synchronize()

    # timed
    t0 = cuda.event()
    t1 = cuda.event()
    t0.record()
    _spmv_kernel[blocks,threads](indptr_d, indices_d, data_d, x_d, y_d)
    t1.record()
    t1.synchronize()
    gpu_time = cuda.event_elapsed_time(t0, t1) / 1e3  # ms→s

    return y_d.copy_to_host(), gpu_time

##############################
# GPU memory (Numba)
##############################
def get_gpu_memory_mb():
    if cuda is None: return None
    info = cuda.current_context().get_memory_info()
    return (info.used // (1024**2))

##############################
# Main
##############################
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spmv","spgemm"], default="spmv")
    p.add_argument("--matrix_path", nargs="+")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--sizeA", type=int, default=1000)
    p.add_argument("--sizeB", type=int, default=1000)
    p.add_argument("--density", type=float, default=0.01)
    p.add_argument("--impl", choices=["naive","gpu"], default="gpu")
    args = p.parse_args()

    # prepare data
    if args.mode=="spmv":
        if args.matrix_path:
            A = mmread(args.matrix_path[0]).tocsr().astype(np.float32)
        else:
            A = sparse_random(args.size,args.size,args.density,format="csr",dtype=np.float32)
        x = np.random.rand(A.shape[1]).astype(np.float32)
    else:
        if args.matrix_path and len(args.matrix_path)==2:
            A = mmread(args.matrix_path[0]).tocsr().astype(np.float32)
            B = mmread(args.matrix_path[1]).tocsr().astype(np.float32)
        else:
            A = sparse_random(args.sizeA,args.sizeB,args.density,format="csr",dtype=np.float32)
            B = sparse_random(args.sizeB,args.sizeA,args.density,format="csr",dtype=np.float32)

    cpu_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    gpu_mem = get_gpu_memory_mb()

    # run
    if args.mode=="spmv":
        # CPU baseline
        t0 = time.time()
        y_cpu = spmv_cpu(A, x)
        cpu_time = time.time()-t0

        if args.impl=="gpu":
            try:
                y_gpu, gpu_time = spmv_numba(A, x)
            except ImportError:
                sys.exit("Numba/CUDA not available")

            diff = np.linalg.norm(y_cpu - y_gpu)
            speed = cpu_time/gpu_time if gpu_time>0 else float("inf")
            print(f"CPU {cpu_time:.4f}s, GPU {gpu_time:.4f}s, speedup {speed:.2f}x, L2 diff {diff:.2e}")
        else:
            gpu_time, diff, speed = 0.0, 0.0, 1.0
            print(f"CPU-only {cpu_time:.4f}s")

        results = [{
            "mode":"spmv","impl":args.impl,
            "cpu_time_sec":cpu_time,"gpu_time_sec":gpu_time,
            "speedup":speed,"l2_norm_diff":diff,
            "cpu_mem_mb":cpu_mem,"gpu_mem_mb":gpu_mem,
            "matrix_size":A.shape[0],"matrix_A_size":0,"matrix_B_size":0,
            "nnz_A":A.nnz,"nnz_B":0
        }]

    else:
        # SpGEMM CPU only for now
        t0 = time.time()
        # call your existing spgemm_cpu here...
        from scipy.sparse import csr_matrix
        C = A.dot(B)
        cpu_time = time.time()-t0
        gpu_time, diff, speed = 0.0, 0.0, 1.0
        print(f"SpGEMM CPU {cpu_time:.4f}s")

        results = [{
            "mode":"spgemm","impl":"naive",
            "cpu_time_sec":cpu_time,"gpu_time_sec":gpu_time,
            "speedup":speed,"l2_norm_diff":diff,
            "cpu_mem_mb":cpu_mem,"gpu_mem_mb":gpu_mem,
            "matrix_size":0,"matrix_A_size":A.shape[0],"matrix_B_size":B.shape[1],
            "nnz_A":A.nnz,"nnz_B":B.nnz
        }]

    log_results_csv(results)

if __name__=="__main__":
    main()
