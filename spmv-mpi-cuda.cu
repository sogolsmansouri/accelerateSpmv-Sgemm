#include <stdio.h>
#include "include/cmdline.h"
#include "include/input.h"
#include "config.h"
#include "include/timer.h"
#include "include/formats.h"
#include <cuda.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "mpi.h"

#define THREADS_PER_BLOCK 1
#define ROOT 0

// #define TESTING

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

typedef struct {
    double gpu_time_sec;    // GPU execution time in seconds
    double speedup;         // CPU time / GPU time
    double l2_norm_diff;    // L2 norm difference between CPU and GPU results
    float cpu_mem_mb;       // CPU memory usage in MB
    float gpu_mem_mb;       // GPU memory usage in MB
    int matrix_size;        // General matrix size (if square)
    int matrix_A_size;      // Size of matrix A (rows)
    int matrix_B_size;      // Size of matrix B (columns)
    int nnz_A;              // Number of non-zeros in A
    int nnz_B;              // Number of non-zeros in B
} benchmark_result_t;

void usage(int argc, char** argv)
{
    printf("Usage: %s [options] [my_matrix.mtx]\n", argv[0]);
    printf("Options:\n");
    printf("  --size N               Generate random matrix of size N x N (default: 1000)\n");
    printf("  --density D            Density of random matrix (0.0 < D < 1.0, default: 0.01)\n");
    printf("  --help                 Display this help message\n");
    printf("Note: If my_matrix.mtx is provided, it must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
    printf("      Otherwise, a random matrix will be generated based on size and density parameters.\n");
}

// Function to generate a random COO matrix
void generate_random_coo_matrix(coo_matrix *coo, int size, double density, int rank)
{
    if (rank == ROOT) {
        printf("Generating random %dx%d COO matrix with density %.4f\n", size, size, density);
    }
    
    // Initialize the matrix structure
    coo->num_rows = size;
    coo->num_cols = size;
    
    // Calculate number of non-zeros based on density
    long long total_elements = (long long)size * size;
    int nnz = (int)(total_elements * density);
    coo->num_nonzeros = nnz;
    
    if (rank == ROOT) {
        printf("Estimated non-zeros: %d\n", nnz);
    }
    
    // Allocate memory
    coo->rows = (int*)malloc(nnz * sizeof(int));
    coo->cols = (int*)malloc(nnz * sizeof(int));
    coo->vals = (float*)malloc(nnz * sizeof(float));
    
    if (!coo->rows || !coo->cols || !coo->vals) {
        printf("Error: Memory allocation failed for random matrix\n");
        exit(1);
    }
    
    // Use a different seed for each run but same across all processes
    srand(42);
    
    // Generate random positions
    for (int i = 0; i < nnz; i++) {
        coo->rows[i] = rand() % size;
        coo->cols[i] = rand() % size;
        coo->vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Random value between -1 and 1
    }
}

__global__ void computation(int *rows, int *cols, float *vals, int nnz, float *x, float *y){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < nnz){
        atomicAdd(&y[rows[idx]], vals[idx] * x[cols[idx]]);
    }
}

double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y, int rank)
{
    int nnz = coo->num_nonzeros;
    // ALLOCATE MEMORY ON DEVICE FOR HOST COPIES OF COO ---------------------------------------
    int *d_rows, *d_cols;
    float *d_vals;
    cudaMalloc((void**)&d_rows, nnz * sizeof(int));
    cudaMalloc((void**)&d_cols, nnz * sizeof(int));
    cudaMalloc((void**)&d_vals, nnz * sizeof(float));
    // Copy host variables to device
    cudaMemcpy(d_rows, coo->rows, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, coo->vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
    // Make host variables point to device memory ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    // DECLARE AND ALLOC DEVICE COPIES OF HOST X and Y ----------------------------------------
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, coo->num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, coo->num_rows * sizeof(float));
    // Copy data from host x and y to device x and y
    cudaMemcpy(d_x, x, coo->num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, coo->num_rows * sizeof(float), cudaMemcpyHostToDevice); // ^^^^^^^^^^^^^^

    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    const int blocks_per_grid = (nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    computation<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_rows, d_cols, d_vals, nnz, d_x, d_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        // Clean up and return
        cudaFree(d_rows);
        cudaFree(d_cols);
        cudaFree(d_vals);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1.0;
    }
    cudaDeviceSynchronize();

    double estimated_time = seconds_elapsed(&time_one_iteration); 

    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0){
        num_iterations = MAX_ITER;
    }
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    if(rank == ROOT){
        printf("\tPerforming %d iterations\n", num_iterations);
    }
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++){
        computation<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_rows, d_cols, d_vals, nnz, d_x, d_y);
    }
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(y, d_y, coo->num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) nnz / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    if(rank == ROOT){
        printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 
    }
    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    
    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    int np, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        MPI_Finalize();
        return 0;
    }

    // Initialize parameters for matrix generation
    int matrix_size = 1000;  // Default size
    double density = 0.01;   // Default density
    char* size_arg = get_arg(argc, argv, "size");
    char* density_arg = get_arg(argc, argv, "density");

    // Parse density argument
    if (density_arg != NULL) {
        density = atof(density_arg);
        if (density <= 0.0 || density >= 1.0) {
            if (rank == ROOT) {
                printf("Error: Density must be between 0 and 1\n");
            }
            MPI_Finalize();
            return -1;
        }
    }

    coo_matrix coo;
    char* mm_filename = NULL;
    bool using_file = false;

    // Generate random matrix
    generate_random_coo_matrix(&coo, matrix_size, density, rank);
    
    if (rank == ROOT) {
        printf("\nMatrix info: rows=%d cols=%d nonzeros=%d\n", 
               coo.num_rows, coo.num_cols, coo.num_nonzeros);
    }
    fflush(stdout);

    if (rank == ROOT) {
        printf("Num Processes: %d\nNum GPU Threads Per Block: %d\n", np, THREADS_PER_BLOCK);
    }

    // Declare and alloc host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));

    // Initialize host copies
    srand(42);  // Use consistent seed for reproducibility
    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }
    for(int i = 0; i < coo.num_rows; i++){
        y[i] = 0;
    }
    
    int *send_counts = (int*)malloc(np * sizeof(int));
    int *displ = (int*)malloc(np * sizeof(int));
    int chunk_size = (coo.num_nonzeros + np - 1) / np;
    int edge = np * chunk_size;
    if(edge > coo.num_nonzeros){
        edge = chunk_size - (edge - coo.num_nonzeros);
        for (int i = 0; i < np; i++) {
            send_counts[i] = chunk_size;
        }
        send_counts[np-1] = edge;
    }else{
        for (int i = 0; i < np; i++) {
            send_counts[i] = chunk_size;
        }
    }
    for(int i = 0; i < np; i++){
        displ[i] = i * chunk_size;
    }

    coo_matrix myCoo;
    myCoo.num_rows = coo.num_rows;
    myCoo.num_cols = coo.num_cols;
    myCoo.num_nonzeros = send_counts[rank];
    myCoo.rows = (int*)malloc(myCoo.num_nonzeros * sizeof(int));
    myCoo.cols = (int*)malloc(myCoo.num_nonzeros * sizeof(int));
    myCoo.vals = (float*)malloc(myCoo.num_nonzeros * sizeof(float));
    float *myY = (float*)malloc(coo.num_rows * sizeof(float));
    memset(myY, 0, coo.num_rows * sizeof(float));

    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(coo.rows, send_counts, displ, MPI_INT, myCoo.rows, send_counts[rank],
         MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(coo.cols, send_counts, displ, MPI_INT, myCoo.cols, send_counts[rank],
         MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(coo.vals, send_counts, displ, MPI_FLOAT, myCoo.vals, send_counts[rank],
         MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    
    double coo_flops = benchmark_coo_spmv(&myCoo, x, myY, rank);
    cudaDeviceSynchronize();
    MPI_Reduce(myY, y, coo.num_rows, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    delete_coo_matrix(&coo);
    free(x);
    free(y);
    free(myCoo.rows);
    free(myCoo.cols);
    free(myCoo.vals);
    free(myY);
    free(send_counts);
    free(displ);
    MPI_Finalize();

    return 0;
}
