#!/bin/bash
# run_all_tests.sh
# Run a suite of tests for both SpMV and SpGEMM in batch

# For SpMV, test different matrix sizes
spmv_sizes=(1000 2000 5000 10000)
density=0.01
impl="gpu"

echo "Running SpMV tests in batch..."
for size in "${spmv_sizes[@]}"; do
    echo "SpMV test: size=${size}"
    python parallel_spmv.py --mode spmv --size "$size" --density "$density" --impl "$impl"
done

# For SpGEMM, test different matrix sizes (square matrices)
spgemm_sizes=(1000 2000 5000)
echo "Running SpGEMM tests in batch..."
for size in "${spgemm_sizes[@]}"; do
    echo "SpGEMM test: sizeA=${size}, sizeB=${size}"
    python parallel_spmv.py --mode spgemm --sizeA "$size" --sizeB "$size" --density "$density" --impl "$impl"
done

echo "All tests completed."
