ARCH ?= sm_35
HOST_COMP ?= mpicc
NVCC = nvcc

OPT_FLAGS = -O0 -g -G

NVCC_FLAGS = -arch=$(ARCH) $(OPT_FLAGS) -ccbin=$(HOST_COMP) --std=c++11

TARGET = mpi_cuda
SOURCES = main.cu cuda_kernels.cu
HEADERS = cuda_kernels.h
LIBS = -lm -lstdc++

P ?= 1
GRID_M ?= 800
GRID_N ?= 1200

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) *.o *.txt

run: $(TARGET)
	bsub -n $(P) \
	     -gpu "num=$(P):mode=exclusive_process" \
	     -R "span[hosts=1]" \
	     -W 02:00 \
	     -o "mpi-cuda-$(GRID_M)-$(GRID_N)-$(P).out" \
	     -e "mpi-cuda-$(GRID_M)-$(GRID_N)-$(P).err" \
	     "mpirun -np $(P) ./$(TARGET) $(GRID_M) $(GRID_N)"

.PHONY: all clean run
