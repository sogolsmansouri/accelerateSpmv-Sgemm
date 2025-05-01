// sp_ops.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <getopt.h>
#include <cuda_runtime.h>
#include <random>

using Timer     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Timer>;

float elapsed_ms(TimePoint a, TimePoint b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

// Stubbed memory‐usage functions (implement as needed)
float get_cpu_memory_mb() { return 0.0f; }
float get_gpu_memory_mb() { return 0.0f; }

// ----------------------------------
// CUDA CSR‐SpMV Kernel
// ----------------------------------
__global__ void csr_spmv_kernel(
    int nrows,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    float sum = 0.0f;
    int start = indptr[row], end = indptr[row+1];
    for (int j = start; j < end; ++j)
        sum += data[j] * x[ indices[j] ];
    y[row] = sum;
}

// ----------------------------------
// CPU SpMV
// ----------------------------------
void spmv_cpu(int nrows,
              const int* indptr,
              const int* indices,
              const float* data,
              const float* x,
              float* y)
{
    for (int i = 0; i < nrows; ++i) {
        float sum = 0.0f;
        for (int j = indptr[i]; j < indptr[i+1]; ++j)
            sum += data[j] * x[ indices[j] ];
        y[i] = sum;
    }
}

// ----------------------------------
// CPU SpGEMM (naive CSR→CSR)
// ----------------------------------
void spgemm_cpu(int A_rows, int A_cols, int B_cols,
                const int* A_indptr,
                const int* A_indices,
                const float* A_data,
                const int* B_indptr,
                const int* B_indices,
                const float* B_data,
                int** C_indptr_out,
                int** C_indices_out,
                float** C_data_out)
{
    std::vector<int> flags(B_cols, -1);
    std::vector<float> temp(B_cols, 0.0f);
    std::vector<int> cols_list;
    cols_list.reserve(B_cols);

    // build C_indptr
    int* C_indptr = (int*)malloc((A_rows+1)*sizeof(int));
    C_indptr[0] = 0;
    for (int i = 0; i < A_rows; ++i) {
        cols_list.clear();
        for (int p = A_indptr[i]; p < A_indptr[i+1]; ++p) {
            int a_col = A_indices[p];
            for (int q = B_indptr[a_col]; q < B_indptr[a_col+1]; ++q) {
                int b_col = B_indices[q];
                if (flags[b_col] != i) {
                    flags[b_col] = i;
                    cols_list.push_back(b_col);
                }
            }
        }
        C_indptr[i+1] = C_indptr[i] + (int)cols_list.size();
    }
    int C_nnz = C_indptr[A_rows];
    int*    C_indices = (int*   )malloc(C_nnz * sizeof(int));
    float*  C_data    = (float* )malloc(C_nnz * sizeof(float));

    std::fill(flags.begin(), flags.end(), -1);
    std::fill(temp.begin(), temp.end(), 0.0f);

    int write_ptr;
    for (int i = 0; i < A_rows; ++i) {
        cols_list.clear();
        for (int p = A_indptr[i]; p < A_indptr[i+1]; ++p) {
            int a_col = A_indices[p];
            float a_val = A_data[p];
            for (int q = B_indptr[a_col]; q < B_indptr[a_col+1]; ++q) {
                int b_col = B_indices[q];
                float b_val = B_data[q];
                if (flags[b_col] != i) {
                    flags[b_col] = i;
                    temp[b_col]  = a_val * b_val;
                    cols_list.push_back(b_col);
                } else {
                    temp[b_col] += a_val * b_val;
                }
            }
        }
        std::sort(cols_list.begin(), cols_list.end());
        write_ptr = C_indptr[i];
        for (int col : cols_list) {
            C_indices[write_ptr] = col;
            C_data   [write_ptr] = temp[col];
            ++write_ptr;
        }
    }

    *C_indptr_out  = C_indptr;
    *C_indices_out = C_indices;
    *C_data_out    = C_data;
}

// ----------------------------------
// Utility: random CSR generator
// ----------------------------------

void generate_random_csr(
    int nrows,
    int ncols,
    float density,
    int** indptr_out,
    int** indices_out,
    float** data_out)
{
    std::mt19937                   gen(123);
    std::uniform_real_distribution<float> prob(0,1), val(0,1);

    // 1) First pass: count nonzeros per row
    std::vector<int> row_nnz(nrows);
    int total_nnz = 0;
    for (int i = 0; i < nrows; ++i) {
        int cnt = 0;
        for (int j = 0; j < ncols; ++j) {
            if (prob(gen) < density) ++cnt;
        }
        row_nnz[i] = cnt;
        total_nnz += cnt;
    }

    // 2) Allocate the CSR arrays
    int* indptr  = (int*)   malloc((nrows+1)*sizeof(int));
    int* indices = (int*)   malloc(total_nnz * sizeof(int));
    float* data  = (float*) malloc(total_nnz * sizeof(float));

    // 3) Build indptr[]
    indptr[0] = 0;
    for (int i = 0; i < nrows; ++i) {
        indptr[i+1] = indptr[i] + row_nnz[i];
    }

    // 4) Fill indices[] and data[]
    int ptr = 0;
    for (int i = 0; i < nrows; ++i) {
        // we want exactly row_nnz[i] entries in this row
        int want = row_nnz[i];
        while (want > 0) {
            int j = gen() % ncols;
            if (prob(gen) < density) {
                indices[ptr] = j;
                data   [ptr] = val(gen);
                ++ptr;
                --want;
            }
        }
        // Optional: you can sort the column indices within each row:
        std::sort(indices + indptr[i],
                  indices + indptr[i+1]);
    }

    *indptr_out  = indptr;
    *indices_out = indices;
    *data_out    = data;
}


// ----------------------------------
// main()
// ----------------------------------
int main(int argc, char** argv)
{
    int mode = 0;   // 1=spmv, 2=spgemm
    int impl = 0;   // 0=cpu, 1=gpu
    int size = 1000, sizeA = 1000, sizeB = 1000;
    float density = 0.01f;

    // parse CLI
    for (int c; (c = getopt(argc, argv, "m:i:s:a:b:d:")) != -1;) {
        switch (c) {
            case 'm': mode    = (strcmp(optarg, "spmv")==0 ? 1 : 2); break;
            case 'i': impl    = (strcmp(optarg, "gpu")==0  ? 1 : 0); break;
            case 's': size    = atoi(optarg);                   break;
            case 'a': sizeA   = atoi(optarg);                   break;
            case 'b': sizeB   = atoi(optarg);                   break;
            case 'd': density = atof(optarg);                   break;
        }
    }

    // Prepare data
    int *A_indptr, *A_indices; float *A_data;
    int *B_indptr, *B_indices; float *B_data;
    if (mode == 1) {
        generate_random_csr(size, size, density, &A_indptr, &A_indices, &A_data);
    } else {
        generate_random_csr(sizeA, sizeB, density, &A_indptr, &A_indices, &A_data);
        generate_random_csr(sizeB, sizeA, density, &B_indptr, &B_indices, &B_data);
    }

    // Vectors for SpMV
    float *x=nullptr, *y=nullptr;
    if (mode == 1) {
        x = (float*)malloc(size*sizeof(float));
        y = (float*)malloc(size*sizeof(float));
        for (int i=0;i<size;++i) x[i] = 1.0f;
    }

    float cpu_ms=0.0f, gpu_ms=0.0f, l2=0.0f, speedup=1.0f;
    // Run the chosen impl
    if (mode == 1) {
        // SpMV
        if (impl==0) {
            auto t0 = Timer::now();
            spmv_cpu(size, A_indptr, A_indices, A_data, x, y);
            auto t1 = Timer::now();
            cpu_ms = elapsed_ms(t0, t1);
        } else {
            // GPU path
            int *d_indptr, *d_indices; float *d_data, *d_x, *d_y;
            int nnz = A_indptr[size];
            cudaMalloc(&d_indptr, (size+1)*sizeof(int));
            cudaMalloc(&d_indices, nnz*sizeof(int));
            cudaMalloc(&d_data,    nnz*sizeof(float));
            cudaMalloc(&d_x,       size*sizeof(float));
            cudaMalloc(&d_y,       size*sizeof(float));

            cudaMemcpy(d_indptr,  A_indptr, (size+1)*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, A_indices, nnz*sizeof(int),     cudaMemcpyHostToDevice);
            cudaMemcpy(d_data,    A_data,    nnz*sizeof(float),   cudaMemcpyHostToDevice);
            cudaMemcpy(d_x,       x,         size*sizeof(float),  cudaMemcpyHostToDevice);

            int threads = 128, blocks = (size + threads -1)/threads;
            cudaDeviceSynchronize();
            cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start);
            csr_spmv_kernel<<<blocks,threads>>>(size, d_indptr, d_indices, d_data, d_x, d_y);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_ms, start, stop);
            gpu_ms /= 1e3f;

            cudaFree(d_indptr); cudaFree(d_indices);
            cudaFree(d_data);    cudaFree(d_x);
            cudaFree(d_y);
        }
        // compute speedup & l2 if you want
        speedup = (gpu_ms>0 ? cpu_ms/gpu_ms : 1.0f);
        l2 = 0.0f;

    } else {
        // SpGEMM CPU only for now
        int *C_indptr,*C_indices; float *C_data;
        auto t0 = Timer::now();
        spgemm_cpu(sizeA,sizeB,sizeA,
                   A_indptr,A_indices,A_data,
                   B_indptr,B_indices,B_data,
                   &C_indptr,&C_indices,&C_data);
        auto t1 = Timer::now();
        cpu_ms = elapsed_ms(t0,t1);
        gpu_ms = 0.0f;
        speedup = 1.0f;
        l2 = 0.0f;
    }

    // sizes & nnz
    int matrix_size    = (mode==1 ? size     : 0);
    int matrix_A_size  = (mode==2 ? sizeA    : 0);
    int matrix_B_size  = (mode==2 ? size      : 0);
    int nnz_A = (mode==1 ? A_indptr[size] : A_indptr[sizeA]);
    int nnz_B = (mode==2 ? B_indptr[sizeA] : 0);

    // print CSV to stdout
    const char* mode_s = (mode==1 ? "spmv" : "spgemm");
    const char* impl_s = (impl==1 ? "gpu"  :
                          (mode==1 ? "naive" : "cpu"));
    printf("%s,%s,%.6f,%.6f,%.2f,%.6e,%.1f,%.1f,%d,%d,%d,%d,%d\n",
        mode_s, impl_s,
        cpu_ms, gpu_ms,
        speedup, l2,
        get_cpu_memory_mb(), get_gpu_memory_mb(),
        matrix_size, matrix_A_size, matrix_B_size,
        nnz_A, nnz_B
    );
    return 0;
}
