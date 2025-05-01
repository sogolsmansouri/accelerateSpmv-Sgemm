
# Accelerated Sparse Operations

This repository demonstrates two implementations of fundamental sparse‐matrix kernels:

- **C++/CUDA** (sp_ops.cu):  
  - CSR-SpMV (GPU via CUDA & CPU fallback)  
  - CSR-SpGEMM (naïve CPU only)

- **Python** (parallel_spmv.py):  
  - SpMV and SpGEMM via SciPy (CPU), multiprocessing (CPU-tiled), and CuPy (GPU)  
  - Automated benchmarking, memory-usage reporting, and CSV logging



## Prerequisites

1. **CUDA Toolkit** (≥10.x) & NVIDIA GPU (compute capability ≥3.0)  
2. **nvcc** (CUDA C++ compiler)  
3. **Python 3.8+**



## Build & Installation

### 1. Clone the repo

git clone https://your.repo.url/accelerated-sparse-ops.git
cd accelerated-sparse-ops


### 2. Build the CUDA executable

nvcc -O3 sp_ops.cu -o sp_ops

This produces the sp_ops binary (for CPU/GPU SpMV and CPU SpGEMM).

### 3. Install Python dependencies

pip install numpy>=1.19.0 scipy>=1.5.0 cupy>=10.0.0
# or, if you prefer a requirements file:
# echo -e "numpy>=1.19.0\nscipy>=1.5.0\ncupy>=10.0.0" > requirements.txt
# pip install -r requirements.txt


> **Note:** For GPU-enabled Python code you may need a CUDA-specific CuPy build, e.g.:  

> pip install cupy-cuda11x




## Usage

### A. C++/CUDA binary (sp_ops)


# SpMV (Sparse Matrix-Vector multiply):
./sp_ops -m spmv -i cpu  -s 1000 -d 0.01
./sp_ops -m spmv -i gpu  -s 1000 -d 0.01

# SpGEMM (Sparse Matrix-Matrix multiply, CPU only):
./sp_ops -m spgemm -i cpu  -a 1000 -b 1000 -d 0.01


- -m : mode (spmv or spgemm)  
- -i : implementation (cpu or gpu for SpMV; cpu for SpGEMM)  
- -s : size (for SpMV square matrix)  
- -a, -b : rows/cols for A and B (SpGEMM)  
- -d : density for random‐matrix generation  

---

### B. Python script (sp_ops_csv.py)

bash
# SpMV via GPU:
python sp_ops_csv.py --mode spmv --size 2000 --density 0.005 --impl gpu

# SpGEMM via tiled CPU:
python sp_ops_csv.py --mode spgemm \
    --sizeA 2000 --sizeB 2000 \
    --density 0.01 --impl parallel --cpus 4

# For full usage help:
python sp_ops_csv.py -h


The script will:
- Generate or read .mtx sparse matrices  
- Run your chosen implementation (naïve/parallel/gpu)  
- Print times, speedups, L₂-norm differences, memory usage  
- Append results to benchmark_results.csv



## Benchmark Results

After running, check benchmark_results.csv for a complete log of:

mode,impl,cpu_time_sec,gpu_time_sec,speedup,l2_norm_diff,
cpu_mem_mb,gpu_mem_mb,matrix_size,matrix_A_size,
matrix_B_size,nnz_A,nnz_B


## Notes

- The C++ memory-usage stubs in sp_ops.cu (get_cpu_memory_mb(), get_gpu_memory_mb()) can be implemented with NVML, getrusage, etc.  
- Adjust Python package versions in the install command to match your environment.

