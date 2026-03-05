#pragma once

#include <stdio.h>

#ifdef NOLOG
#define FLASHLOG(...)
#else
#define FLASHLOG(fmt, ...)                   \
  {                                           \
    printf("AllToAll LOG: " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout);                       \
  }
#endif

#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)

#define SCALING_FACTOR_MIN 1
#define SCALING_FACTOR_MAX 1000000
#define SCALING_FACTOR_STEP 2

#define MAX_SUM_LIMIT 100
#define BENCHMARK_DIR "benchmark/"


typedef struct balance_data_t* BalancePtr;
typedef uint TransferMatrixElement;
typedef struct recv_data_t * RestorePtr;
typedef struct recv_data_t * DirectCpyPtr;
typedef uint * ChannelPtr;

#define MAX_ELEMENT_NUM 3

#define GPU_NUM_PER_SERVER 8
#define MAX_SERVER_NUM 10
#define MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER 80
#define MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE 640
#define MAX_SERVER_NUM_DOUBLE 20
#define MAX_SERVER_NUM_SQUARE 100
#define MAX_TRANSFER_STEP_NUM 101 // MAX_SERVER_NUM_SQUARE + 1
#define MAX_GPU_PER_SERVER GPU_NUM_PER_SERVER
#define MAX_GPU_PER_SERVER_SQUARE 64

#define mem_512B_align(x) (x + 0x1ff) & 0xfffffffffffffe00
#define mem_align(x) mem_512B_align(x)

#define ABLATION_TEST 1
#define ROCEV2_PAYLOAD_TPUT 11   //tested via ib_send_bw; 100Gbps NIC