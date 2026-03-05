.PNONY: flash compile run clean debug

ROCM_PATH = /opt/rocm
TARGET = all2all_tester
HIPCC = $(ROCM_PATH)/bin/hipcc
MPI_HOME = /opt/ompi
CFLAGS = -O3 -std=c++17
RCCL_HOME ?= /path/to/rccl/build
LIBTORCH_HOME ?= /path/to/libtorch

INC =  -I$(RCCL_HOME)/ -I$(RCCL_HOME)/include -I$(RCCL_HOME)/hipify/src/include
INC += -I$(ROCM_PATH)/include
INC += -I$(ROCM_PATH)/include/hip
INC += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi
INC += -I./src/include
INC += -I$(LIBTORCH_HOME)/include/torch/csrc/api/include
INC += -I$(LIBTORCH_HOME)/include

BUILD_DIR = $(shell pwd)/build


LIB = -lpthread  -L$(RCCL_HOME) -L$(RCCL_HOME)/lib
LIB += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
LIB += -L${MPI_HOME}/lib -lmpi
LIB += -Wl,-rpath,$(RCCL_HOME) -L$(RCCL_HOME) -lrccl
LIB += -L$(LIBTORCH_HOME)/lib -ltorch -ltorch_hip -lc10 -lc10_hip -ltorch_python

SOURCES = test_alltoall.cpp
SRC_DIR = ./src/fast_alltoall
INC_DIR = ./src/include/fast_alltoall

OBJ = $(BUILD_DIR)/alltoall_matrix.o
OBJ += $(BUILD_DIR)/alltoall_algorithm.o
OBJ += $(BUILD_DIR)/alltoall_local_scheduler.o
OBJ += $(BUILD_DIR)/alltoall_global_scheduler.o

flash: flash.cpp $(INC_DIR)/alltoall_define.h $(BUILD_DIR)/alltoall_matrix.o $(BUILD_DIR)/alltoall_algorithm.o $(BUILD_DIR)/alltoall_local_scheduler.o $(BUILD_DIR)/alltoall_global_scheduler.o
	@echo "Building FLASH AllToAll"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(HIPCC) $(CFLAGS) $(INC) $(OBJ) -fPIC $(shell python3 -m pybind11 --includes) -shared flash.cpp  -o FlashAllToAll$(shell python3-config --extension-suffix) $(LIB)
	@echo "Building successful"


compile: test_alltoall.cpp $(INC_DIR)/alltoall_define.h $(BUILD_DIR)/alltoall_matrix.o $(BUILD_DIR)/alltoall_algorithm.o $(BUILD_DIR)/alltoall_local_scheduler.o $(BUILD_DIR)/alltoall_global_scheduler.o
	@echo "Building AllToAll Tester"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(HIPCC) $(CFLAGS) $(INC) $(OBJ) $(SOURCES) -o $(BUILD_DIR)/$(TARGET) $(LIB)
	@echo "Building successful"

$(BUILD_DIR)/alltoall_matrix.o: ${SRC_DIR}/alltoall_matrix.cc $(INC_DIR)/alltoall_matrix.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_matrix.cc -fPIC  -o $(BUILD_DIR)/alltoall_matrix.o

$(BUILD_DIR)/alltoall_algorithm.o: ${SRC_DIR}/alltoall_algorithm.cc $(INC_DIR)/alltoall_algorithm.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_algorithm.cc -fPIC  -o $(BUILD_DIR)/alltoall_algorithm.o

$(BUILD_DIR)/alltoall_local_scheduler.o: ${SRC_DIR}/alltoall_local_scheduler.cc $(INC_DIR)/alltoall_local_scheduler.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_local_scheduler.cc -fPIC  -o $(BUILD_DIR)/alltoall_local_scheduler.o

$(BUILD_DIR)/alltoall_global_scheduler.o: ${SRC_DIR}/alltoall_global_scheduler.cc $(INC_DIR)/alltoall_global_scheduler.h  $(INC_DIR)/alltoall_define.h
	$(HIPCC) $(CFLAGS) $(INC) -c ${SRC_DIR}/alltoall_global_scheduler.cc -fPIC  -o $(BUILD_DIR)/alltoall_global_scheduler.o


run:
	mpirun --allow-run-as-root -hostfile <path-to-hostfile> -map-by ppr:8:node --bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include <net-iface> -x LD_LIBRARY_PATH=$(ROCM_PATH)/lib:$LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=<net-iface> -x LD_PRELOAD=<path-to-rccl-lib>:$LD_PRELOAD -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,GRAPH -x HSA_FORCE_FINE_GRAIN_PCIE=1 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_NET_GDR_LEVEL=3 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_IBEXT_DISABLE=1 -x NCCL_PROTO=Simple bash run.sh

clean:
	@echo "Cleaning FastAll2All"
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

debug:
	sudo gdb --args $(BUILD_DIR)/$(TARGET)
