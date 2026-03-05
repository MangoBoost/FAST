#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef unsigned int uint;

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
#define GPU_NUM_PER_SERVER_SQUARE 64
#define GPU_NUM_PER_SERVER_TIMES_MAX_BLOCK_NUM 256 // 8 * 32
#define GPU_NUM_PER_SERVER_SQUARE_TIMES_MAX_BLOCK_NUM 2048 // 8 * 8 * 32
#define MAX_SERVER_NUM 10
#define MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER 80
#define MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE 640
#define MAX_SERVER_NUM_DOUBLE 20
#define MAX_SERVER_NUM_SQUARE 100
#define MAX_TRANSFER_STEP_NUM 101 // MAX_SERVER_NUM_SQUARE + 1
#define MAX_GPU_PER_SERVER GPU_NUM_PER_SERVER
#define MAX_GPU_PER_SERVER_SQUARE 64

#define mem_512B_align(x) ((x + 0x1ff) & 0xfffffffffffffe00ULL)
#define mem_align(x) mem_512B_align(x)

#define ABLATION_TEST 1
#define ETHERNET_100G_PAYLOAD_1KMTU_TPUT 11 // GBps
#define ETHERNET_400G_PAYLOAD_4KMTU_TPUT 48.85 // GBps
#define INTER_SERVER_LINK_TPUT ETHERNET_400G_PAYLOAD_4KMTU_TPUT

#define H200_NVLINK_TPUT 380  // GBps, duplex
#define INTRA_SERVER_LINK_TPUT H200_NVLINK_TPUT

#define FLASH_PROFILE 1

// set platform
#define CUDA_NCCL_COMPILE 1
#define ROCM_RCCL_COMPILE 0

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y) (((x + y - 1) / y) * y)
#define TX_UNROLL_FACTOR 4
#define THREAD_N_PER_WARP 32
#define THREAD_N_PER_2WARP 64

// pipeline trunk size
#define FLASH_MAX_CHUNK_NUM 16
#define FLASH_MIN_CHUNK_SIZE (32 << 20) //8 MB, in unit of byte
#define FLASH_4MB_ALIGN(x) ((x + 0x3fffff) & 0xffffffffffc00000ULL)
#define FLASH_1MB_ALIGN(x) ((x + 0xfffff) & 0xfffffffffff00000ULL)
#define FLASH_CHUNK_ALIGN(x) FLASH_4MB_ALIGN(x)

#define INTRANODE_RECV_COMPLETION_SIGNAL_LENGTH 32