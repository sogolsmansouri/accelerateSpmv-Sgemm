Martin Carignan
CSC 548 - Spring 2025

Python SpMV MPI
-------------------------------------------------------------
NOTE: To run this function you need to activate your python virtual environment for it to work properly then run the spmv-mpi.batch script, the outputs will be in a folder called output/.

This function performs Sparse Matrix-Vector Multiplication (SpMV) using MPI-based parallelization(mpi4py) across CPU processes. It accepts a sparse matrix A in Compressed Sparse Row (CSR) format and a dense vector x, and returns the result of the multiplication y = Ax.

With MPI it distributes matrix rows across available MPI processes.It does this be dynamically partitioning rows by calculating how many rows from the matrix the thread will work on:

    rows_per_proc = m // size
    remainder = m % size
    start_row = rank * rows_per_proc + min(rank, remainder)
    end_row = start_row + rows_per_proc + (1 if rank < remainder else 0)
    local_rows = end_row - start_row

This ensures load balancing by accounting for uneven row counts. This is very important because we broadcasting the CSR matrix A and vector x are from the root to all processes. After that each process computes its portion of the output vector y based on its assigned rows.Finally, all local results are gathered at the root process using MPI.Gatherv.

C SpMV MPI-CUDA
-------------------------------------------------------------
    NOTE: To test different sizes you need to manually change the size in the source code at line 196 in spmv-mpi-cuda.cu and then you can run the spmv-mpi-cuda.batch script, the output will be in a folder called outputs_mpi_4/.

    This program performs Sparse Matrix-Vector Multiplication (SpMV) using a randomly generated COO-format sparse matrix on the C language, parallelized across multiple MPI processes and accelerated with CUDA on GPUs. Originally we palnned to use CuPy+MPI but at the point when we reached this step in our project, we had issues with the cluster backend that seemed to not get CuPy to work, so we tried to imitate it with MPI+CUDA. This uses my HW3 implementation of MPI+CUDA but modified by adding a function to generate a random coo matrix.

    generate_random_coo_matrix() creates a square sparse matrix of a given size and density, with values in [-1, 1]. I ran into issue compiling the program a certain way so hardcoding the sizes in to the program was how I ran it through different iterations of sizes.

    For distribution, matrix nonzeros are evenly partitioned across MPI processes using MPI_Scatterv.

    The CUDA kernel performs y[rows[i]] += vals[i] * x[cols[i]] using atomic operations.

    To benchmark we use benchmark_coo_spmv() runs the SpMV kernel multiple times to measure runtime, GFLOPS, and bandwidth.

    Each process computes partial results on GPU and uses MPI_Reduce to combine the final output on root.