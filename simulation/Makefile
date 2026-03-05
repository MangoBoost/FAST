.PNONY: build run clean debug plot

TARGET = all2all_simulator
CC = g++
CFLAGS = -O3 -std=c++11
TEST_DIR = $(shell pwd)
BUILD_DIR = $(shell pwd)/build
ALGO_DIR = $(shell pwd)/algorithm
SCHE_DIR = $(shell pwd)/scheduler

INCLUDES = -I$(shell pwd)
INCLUDES += -I$(ALGO_DIR)
INCLUDES += -I$(SCHE_DIR)

OBJS = $(BUILD_DIR)/matrix.o
OBJS += $(BUILD_DIR)/all2all.o
OBJS += $(BUILD_DIR)/test.o
OBJS += $(BUILD_DIR)/config.o
OBJS += $(BUILD_DIR)/local.o
OBJS += $(BUILD_DIR)/global.o


LD = -lpthread

SOURCES = main.cpp

build: main.cpp define.h $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(BUILD_DIR)/test.o $(BUILD_DIR)/local.o $(BUILD_DIR)/global.o $(BUILD_DIR)/config.o
	@echo "Building FastAll2All"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(CC) -g $(CFLAGS) $(INCLUDES) $(OBJS) $(SOURCES) -o $(BUILD_DIR)/$(TARGET) $(LD)
	@echo "Building successful"

$(BUILD_DIR)/all2all.o: $(BUILD_DIR)/matrix.o ${ALGO_DIR}/all2all.cpp $(ALGO_DIR)/all2all.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c ${ALGO_DIR}/all2all.cpp -o $(BUILD_DIR)/all2all.o

$(BUILD_DIR)/matrix.o: ${ALGO_DIR}/matrix.cpp $(ALGO_DIR)/matrix.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c ${ALGO_DIR}/matrix.cpp -o $(BUILD_DIR)/matrix.o

$(BUILD_DIR)/test.o: $(BUILD_DIR)/all2all.o $(BUILD_DIR)/matrix.o test.cpp test.h define.h $(BUILD_DIR)/config.o $(BUILD_DIR)/local.o $(BUILD_DIR)/global.o
	$(CC) $(CFLAGS) $(INCLUDES) -c test.cpp -o $(BUILD_DIR)/test.o

$(BUILD_DIR)/config.o: $(SCHE_DIR)/config.cpp $(SCHE_DIR)/config.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/config.cpp -o $(BUILD_DIR)/config.o

$(BUILD_DIR)/local.o: $(BUILD_DIR)/config.o $(SCHE_DIR)/local.cpp $(SCHE_DIR)/local.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/local.cpp -o $(BUILD_DIR)/local.o

$(BUILD_DIR)/global.o: $(BUILD_DIR)/config.o  $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(SCHE_DIR)/global.cpp $(SCHE_DIR)/global.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/global.cpp -o $(BUILD_DIR)/global.o

run:
	$(BUILD_DIR)/$(TARGET)

clean:
	@echo "Cleaning FastAll2All"
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

debug:
	sudo gdb --args $(BUILD_DIR)/$(TARGET)

PLOT_DIR := $(shell pwd)/plot
plot: FORCE
	$(shell cd plot && python3 plot_server_gpu_number.py && python3 plot_intra_speed_skewness.py)

FORCE: ;