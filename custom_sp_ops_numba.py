#!/usr/bin/env python3
"""
Custom Numba-accelerated sparse operations:
  - spmv_numba: CSR sparse matrix-vector multiplication (parallel)
  - spgemm_numba: CSR sparse matrix-matrix multiplication (parallel by rows)
"""
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange

@njit(parallel=True)
def spmv_numba_parallel(indptr, indices, data, x, m):
    """Parallel CSR SpMV: y = A * x"""
    y = np.zeros(m, dtype=np.float32)
    for i in prange(m):
        row_start = indptr[i]
        row_end = indptr[i+1]
        total = 0.0
        for idx in range(row_start, row_end):
            total += data[idx] * x[indices[idx]]
        y[i] = total
    return y

def spmv_numba(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    """Numba-accelerated CSR SpMV"""
    A = A.tocsr()
    m = A.shape[0]
    return spmv_numba_parallel(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float32),
        x.astype(np.float32),
        m
    )

@njit(parallel=True)
def spgemm_numba_parallel(indptr_a, indices_a, data_a,
                          indptr_b, indices_b, data_b,
                          m, n_cols_b):
    """
    Parallel CSR SpGEMM by rows: each row of C is computed in parallel.
    """
    # We'll accumulate each row in a Python dict-like structure,
    # then flatten into CSR arrays. This is illustrative.
    import scipy.sparse as sp  # only for typing
    data_list = []
    indices_list = []
    indptr = np.zeros(m+1, dtype=np.int32)
    nnz = 0

    for i in prange(m):
        # accumulate row i in a small Python dict
        row_dict = {}
        for idx in range(indptr_a[i], indptr_a[i+1]):
            a_col = indices_a[idx]
            a_val = data_a[idx]
            for j in range(indptr_b[a_col], indptr_b[a_col+1]):
                b_col = indices_b[j]
                row_dict[b_col] = row_dict.get(b_col, 0.0) + a_val * data_b[j]
        # write this row
        cols = sorted(row_dict.keys())
        for c in cols:
            indices_list.append(c)
            data_list.append(row_dict[c])
        nnz += len(cols)
        indptr[i+1] = nnz

    return (np.array(data_list, dtype=np.float32),
            np.array(indices_list, dtype=np.int32),
            indptr)

def spgemm_numba(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    """Numba-accelerated CSR SpGEMM"""
    A = A.tocsr(); B = B.tocsr()
    m, n_cols_b = A.shape[0], B.shape[1]
    data, indices, indptr = spgemm_numba_parallel(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float32),
        B.indptr.astype(np.int32),
        B.indices.astype(np.int32),
        B.data.astype(np.float32),
        m, n_cols_b
    )
    return csr_matrix((data, indices, indptr), shape=(m, n_cols_b))
