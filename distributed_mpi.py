#!/usr/bin/env python3
"""
sp_ops_all.py

Accelerated sparse operations in one file:
  - naive CPU
  - numba‐parallel CPU
  - CuPy RawKernel GPU
  - MPI‐distributed (numba or gpu)

Usage examples:
  # naive SpMV
  python sp_ops_all.py --mode spmv   --impl naive   --size 1000 --density 0.01

  # numba SpGEMM
  python sp_ops_all.py --mode spgemm --impl numba   --sizeA 500 --sizeB 500 --density 0.02

  # GPU RawKernel SpMV
  python sp_ops_all.py --mode spmv   --impl gpu     --size 2000 --density 0.005

  # MPI‐distributed SpMV (4 ranks)
  mpirun -n 4 python sp_ops_all.py --mode spmv --impl mpi --matrix matrix.mtx --gpu
"""

import argparse, time, csv, resource, sys
import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random
from scipy.io import mmread

# Try Numba
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

# Try CuPy
try:
    import cupy as cp
    from cupyx import scipy as cpx
    GPU_OK = True
except ImportError:
    GPU_OK = False

# Try MPI
try:
    from mpi4py import MPI
    MPI_OK = True
except ImportError:
    MPI_OK = False

# -----------------------------------------------------------------------------
# 1) Naive CPU
# -----------------------------------------------------------------------------
def spmv_cpu(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    A = A.tocsr()
    m = A.shape[0]
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        for idx in range(A.indptr[i], A.indptr[i+1]):
            y[i] += A.data[idx] * x[A.indices[idx]]
    return y

def spgemm_cpu(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    return A.dot(B)

# -----------------------------------------------------------------------------
# 2) Numba‐parallel CPU
# -----------------------------------------------------------------------------
if NUMBA_OK:
    @njit(parallel=True)
    def _spmv_nb(indptr, indices, data, x, m):
        y = np.zeros(m, dtype=np.float32)
        for i in prange(m):
            total = 0.0
            for j in range(indptr[i], indptr[i+1]):
                total += data[j] * x[indices[j]]
            y[i] = total
        return y

    def spmv_numba(A, x):
        A = A.tocsr()
        return _spmv_nb(A.indptr.astype(np.int32),
                        A.indices.astype(np.int32),
                        A.data.astype(np.float32),
                        x.astype(np.float32),
                        A.shape[0])

    @njit(parallel=True)
    def _spgemm_nb(ip_a, idx_a, dat_a, ip_b, idx_b, dat_b, m, n_b):
        # accumulate per‐row in Python dict, then flatten (illustrative)
        data_list = []
        idx_list  = []
        indptr    = np.zeros(m+1, dtype=np.int32)
        nnz = 0
        for i in prange(m):
            row = {}
            for ja in range(ip_a[i], ip_a[i+1]):
                a_col = idx_a[ja]; a_val = dat_a[ja]
                for jb in range(ip_b[a_col], ip_b[a_col+1]):
                    b_col = idx_b[jb]; b_val = dat_b[jb]
                    row[b_col] = row.get(b_col, 0.0) + a_val*b_val
            cols = sorted(row.keys())
            for c in cols:
                idx_list.append(c)
                data_list.append(row[c])
            nnz += len(cols)
            indptr[i+1] = nnz
        return (np.array(data_list, dtype=np.float32),
                np.array(idx_list,  dtype=np.int32),
                indptr)

    def spgemm_numba(A, B):
        A = A.tocsr(); B = B.tocsr()
        m, n_b = A.shape[0], B.shape[1]
        data, idxs, ip = _spgemm_nb(A.indptr.astype(np.int32),
                                   A.indices.astype(np.int32),
                                   A.data.astype(np.float32),
                                   B.indptr.astype(np.int32),
                                   B.indices.astype(np.int32),
                                   B.data.astype(np.float32),
                                   m, n_b)
        return csr_matrix((data, idxs, ip), shape=(m, n_b))
else:
    def spmv_numba(*args, **kwargs): raise RuntimeError("Numba not installed")
    def spgemm_numba(*args, **kwargs): raise RuntimeError("Numba not installed")

# -----------------------------------------------------------------------------
# 3) CuPy RawKernel GPU SpMV
# -----------------------------------------------------------------------------
if GPU_OK:
    _spmv_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void spmv(int m,
              const int* ip,
              const int* idx,
              const float* dat,
              const float* x,
              float* y) {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i<m){
            float s=0;
            for(int j=ip[i];j<ip[i+1];j++)
                s += dat[j]*x[idx[j]];
            y[i]=s;
        }
    }
    ''','spmv')

    def spmv_gpu(A, x):
        A = A.tocsr()
        m = A.shape[0]
        ip  = cp.asarray(A.indptr, dtype=cp.int32)
        idx = cp.asarray(A.indices, dtype=cp.int32)
        dat = cp.asarray(A.data,    dtype=cp.float32)
        xg  = cp.asarray(x,         dtype=cp.float32)
        yg  = cp.zeros(m, dtype=cp.float32)
        threads = 256
        blocks  = (m+threads-1)//threads
        _spmv_kernel((blocks,),(threads,),
                     (m, ip, idx, dat, xg, yg))
        cp.cuda.Stream.null.synchronize()
        return yg.get()
else:
    def spmv_gpu(*args, **kwargs): raise RuntimeError("CuPy not installed")

# -----------------------------------------------------------------------------
# 4) MPI‐distributed
# -----------------------------------------------------------------------------
if MPI_OK:
    def _scatter(A, comm):
        rank, size = comm.Get_rank(), comm.Get_size()
        m = A.shape[0]
        counts = [(m//size)+(1 if i<m%size else 0) for i in range(size)]
        starts = [sum(counts[:i]) for i in range(size)]
        return A[starts[rank]:starts[rank]+counts[rank]].copy()

    def distributed_spmv(A, x, use_gpu, comm):
        # broadcast A and x
        A = comm.bcast(A if comm.Get_rank()==0 else None, root=0)
        x = comm.bcast(x if comm.Get_rank()==0 else None, root=0)
        # scatter rows
        A_loc = _scatter(A, comm)
        # local compute
        y_loc = spmv_gpu(A_loc, x) if (use_gpu and GPU_OK) else spmv_numba(A_loc,x) if NUMBA_OK else spmv_cpu(A_loc,x)
        # gather
        ys = comm.gather(y_loc, root=0)
        if comm.Get_rank()==0:
            return np.concatenate(ys)
        return None

    def distributed_spgemm(A, B, comm):
        A = comm.bcast(A if comm.Get_rank()==0 else None, root=0)
        B = comm.bcast(B if comm.Get_rank()==0 else None, root=0)
        A_loc = _scatter(A, comm)
        C_loc = spgemm_numba(A_loc, B) if NUMBA_OK else spgemm_cpu(A_loc,B)
        parts = comm.gather((C_loc.indptr, C_loc.indices, C_loc.data), root=0)
        if comm.Get_rank()!=0: return None
        # merge CSR blocks
        indptr=[0]; idxs=[]; dat=[]
        offset=0
        for i,(ip,ids,dt) in enumerate(parts):
            block_ip = ip if i==0 else ip[1:]
            for v in block_ip:
                indptr.append(v+offset)
            offset = indptr[-1]
            idxs.extend(ids.tolist()); dat.extend(dt.tolist())
        return csr_matrix((np.array(dat),np.array(idxs),np.array(indptr)),shape=(A.shape[0],B.shape[1]))
else:
    distributed_spmv = distributed_spgemm = None

# -----------------------------------------------------------------------------
# Helpers & main
# -----------------------------------------------------------------------------
def get_cpu_mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
def get_gpu_mem():
    try: return cp.get_default_memory_pool().used_bytes()/(1024**2)
    except: return 0

def log_csv(r,fn="benchmark_results.csv"):
    fld = ["mode","impl","cpu_t","gpu_t","speedup","l2","cpu_mem","gpu_mem","A_n","B_n","ranks"]
    hdr = not open(fn, "a").readline()
    with open(fn,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fld)
        if hdr: w.writeheader()
        w.writerow(r)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spmv","spgemm"], required=True)
    p.add_argument("--impl",choices=["naive","numba","gpu","mpi"],required=True)
    p.add_argument("--matrix",nargs="+")
    p.add_argument("--size", type=int,    default=1000)
    p.add_argument("--sizeA",type=int,    default=1000)
    p.add_argument("--sizeB",type=int,    default=1000)
    p.add_argument("--density",type=float, default=0.01)
    p.add_argument("--gpu",action="store_true")
    args=p.parse_args()

    # prepare data
    if args.mode=="spmv":
        if args.matrix:
            A=mmread(args.matrix[0]).tocsr().astype(np.float32)
        else:
            A=sparse_random(args.size,args.size,density=args.density,format="csr",dtype=np.float32,random_state=123)
        x=np.random.rand(A.shape[1]).astype(np.float32)
    else:
        if args.matrix and len(args.matrix)>=2:
            A=mmread(args.matrix[0]).tocsr().astype(np.float32)
            B=mmread(args.matrix[1]).tocsr().astype(np.float32)
        else:
            A=sparse_random(args.sizeA,args.sizeB,density=args.density,format="csr",dtype=np.float32,random_state=123)
            B=sparse_random(args.sizeB,args.sizeA,density=args.density,format="csr",dtype=np.float32,random_state=123)

    cpu_mem0=get_cpu_mem(); gpu_mem0=get_gpu_mem();
    cpu_t=gpu_t=l2=speedup=0; ranks=1

    # run and time
    if args.impl=="naive":
        t0=time.time(); 
        if args.mode=="spmv": y=spmv_cpu(A,x)
        else:                C=spgemm_cpu(A,B)
        cpu_t=time.time()-t0
    elif args.impl=="numba":
        if not NUMBA_OK: sys.exit("Numba missing")
        t0=time.time()
        if args.mode=="spmv": y=spmv_numba(A,x)
        else:                C=spgemm_numba(A,B)
        cpu_t=time.time()-t0
        if args.mode=="spmv": 
            y0=spmv_cpu(A,x); l2=np.linalg.norm(y0-y)
        else:
            C0=spgemm_cpu(A,B); l2=np.linalg.norm((C0-C).toarray())
    elif args.impl=="gpu":
        if not GPU_OK: sys.exit("CuPy missing")
        if args.mode!="spmv": sys.exit("GPU only for spmv")
        t0=time.time(); yg=spmv_gpu(A,x); gpu_t=time.time()-t0
        t0=time.time(); yc=spmv_cpu(A,x); cpu_t=time.time()-t0
        l2=np.linalg.norm(yc-yg); speedup=cpu_t/gpu_t
    else:  # mpi
        if not MPI_OK: sys.exit("MPI missing")
        comm=MPI.COMM_WORLD; ranks=comm.Get_size()
        t0=time.time()
        res=distributed_spmv(A,x,args.gpu,comm) if args.mode=="spmv" else distributed_spgemm(A,B,comm)
        if comm.Get_rank()==0:
            cpu_t=time.time()-t0
            if args.mode=="spmv":
                y0=spmv_cpu(A,x); l2=np.linalg.norm(y0-res)
            else:
                C0=spgemm_cpu(A,B); l2=np.linalg.norm((C0-res).toarray())

    # report
    if args.impl!="mpi" or MPI_OK and MPI.COMM_WORLD.Get_rank()==0:
        print(f"{args.impl=} {args.mode=} cpu_t={cpu_t:.4f}s gpu_t={gpu_t:.4f}s speedup={speedup:.2f} l2={l2:.2e}")
        print(f"mem CPU {cpu_mem0:.1f}MB GPU {gpu_mem0:.1f}MB ranks={ranks}")
        log_csv({
            "mode":args.mode,"impl":args.impl,
            "cpu_t":cpu_t,"gpu_t":gpu_t,"speedup":speedup,"l2":l2,
            "cpu_mem":cpu_mem0,"gpu_mem":gpu_mem0,
            "A_n":A.nnz, "B_n":0 if args.mode=="spmv" else B.nnz,
            "ranks":ranks
        })

if __name__=="__main__":
    main()
