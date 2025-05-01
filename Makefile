NVCC = nvcc
NVFLAGS = -O3 -std=c++11 -I./include/ -diag-suppress 177,2464
CC=gcc
OBJS=mmio.o
LDFLAG=-O3
MPI_CC=mpicc
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings

all: spmv spmv-cuda spmv-mpi-cuda

spmv-cuda: spmv-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-cuda.cu -o $@

spmv-mpi-cuda: spmv-mpi-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin mpicxx -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi

spmv-mpi-opt: spmv-mpi-opt.o ${OBJS}
	${MPI_CC} ${LDFLAG} -o $@ $^

spmv: spmv.o ${OBJS}
	${CC} ${LDFLAG} -o $@ $^

clean:
	rm -f spmv-cuda spmv-mpi-cuda mmio.o
