#pragma once 

#include <iostream>
#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>
#include <cuda_runtime.h>
#include "alltoall_global_scheduler.h"

struct kernel_sz_t{
    dim3 grid_dim;
    dim3 block_dim;
    uint32_t warp_n_per_block;
    uint64_t block_n;
    uint64_t min_tx_sz_at_a_time;
};

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define NVSHMEMCHECK(stmt)                                                                         \
  do {                                                                                             \
    int result = (stmt);                                                                           \
    if (NVSHMEMX_SUCCESS != result) {                                                              \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result);      \
      exit(-1);                                                                                    \
    }                                                                                              \
  } while (0)

#define CUCHECK(cmd) do {                                                                          \
  CUresult res = cmd;                                                                              \
  if (res != CUDA_SUCCESS) {                                                                       \
    fprintf(stderr, "[%s:%d] cu failed with error %d \n", __FILE__, __LINE__, res);                \
    exit(1);                                                                                       \
  }                                                                                                \
} while(0)


struct kernel_sz_t kernel_sz_for_workload(uint warp_n_per_block, uint block_n, uint TX_BURST);

inline int get_sm_count() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  int numSMs;
  CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
  return numSMs;
}


struct flash_nvshmem_buffer_t{
    uint8_t * send_buffer;
    uint8_t * recv_buffer[2];   // recv ping-pong buffer
    uint8_t * load_balance_send_buffer;
    uint8_t * load_balance_recv_buffer;
    uint8_t * redistribute_recv_buffer;
    uint64_t * sync_signal;     // ping-pong
};



struct shared_memory_t{
  void * local_data;
  void ** remote_data;
  void ** host_remote_data;
  cudaIpcMemHandle_t local_handle;
  cudaIpcMemHandle_t * remote_handle;
};

struct shared_memory_t init_share_memory(void * dev_mem_ptr, uint rank_n_per_node);
void free_share_memory(struct shared_memory_t  * smem); 

struct alltoallv_shared_memory_t{
  struct shared_memory_t recv_buffer;
};

#define VERIFY_BUFFER 1
struct fanout_data_buffer_t{
  void * buffer_in;     // originial send buffer
  uint64_t in_total_sz;
  uint64_t * in_sz_per_rank;
  uint64_t * in_disp_per_rank;
  void * buffer_out;    // final recv buffer
  uint64_t out_total_sz;
  uint64_t * out_sz_per_rank;
  uint64_t * out_disp_per_rank;
  struct alltoallv_shared_memory_t share_mem;
#if VERIFY_BUFFER == 1
  void * buffer_verify;
#endif
};

#define VERIFY_BUFFER 1
struct intra_then_inter_data_buffer_t{
// ------------------------------
 // |         IN BUFFER         | - in_total_sz
 // | INTRA RANK   |  INTER RANK|
 // | RANK 0 | RANK 1 | RANK 2 | ... |
// ------------------------------

  void * buffer_in;     // originial send buffer
  uint64_t in_total_sz;
  uint64_t * in_sz_per_rank;
  uint64_t * in_disp_per_rank;  //for each dst rank, disp at local send buffer
  void * buffer_out;    // final recv buffer
  uint64_t out_total_sz;
  uint64_t * out_sz_per_rank;
  uint64_t * out_disp_per_rank; // for each src rank, disp at local recv buffer
  uint64_t * out_disp_at_remote_buffer; // for dst ranks, the offset in the remote recv buffer
  struct alltoallv_shared_memory_t share_mem;
  uint64_t intra_send_sz;
  uint64_t intra_recv_sz;
#if VERIFY_BUFFER == 1
  void * buffer_verify;
#endif
};

struct intra_then_inter_data_buffer_t init_data_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n);
void free_data_buffer(struct intra_then_inter_data_buffer_t * buf);

struct fanout_nvshmem_buffer_t{
    uint8_t * send_buffer;
    uint8_t * recv_buffer;
    uint64_t * sync_signal;
};

// for GPU use
struct fanout_buffer_params_t{
  uint64_t * sender_send_disp; // rank_n * block_n
  uint64_t * sender_transfer_sz;   // rank_n * block_n
  uint64_t * sender_recv_disp; // rank_n * block_n
  uint64_t * recver_transfer_sz; // rank_n * block_n
  uint64_t * recver_recv_disp; // rank_n * block_n
  uint64_t * intra_round_n;    // local_rank_n * block_n, sending round# for intra-node alltoall
  uint64_t * inter_send_round_n;
  uint64_t * inter_recv_round_n;
};

struct internode_transfer_params_t{
  uint64_t * sender_send_disp; // rank_n
  uint64_t * sender_transfer_sz;  // rank_n
  uint64_t * sender_recv_disp; // rank_n
  uint64_t * recver_transfer_sz; // rank_n
};

struct intranode_transfer_params_t{
  uint64_t * sender_send_disp; // local_rank_n * block_n
  uint64_t * sender_transfer_sz;  // local_rank_n * block_n
  uint64_t * sender_recv_disp; // local_rank_n * block_n
  uint64_t * recver_transfer_sz; // local_rank_n * block_n
  uint64_t * recver_recv_disp; // local_rank_n * block_n
  uint64_t * intra_round_n;    // local_rank_n * block_n, sending round# for intra-node alltoall
};


struct flash_shared_memory_t{
  struct shared_memory_t out_buffer;   // output buffer
  struct shared_memory_t balance_recv_buffer;   // buffer for receiving load-balanced data
  struct shared_memory_t recv_complete_signal;
};

// buffer for FLASH scheduler
struct flash_buffer_ptr_t{
  void * buffer_in;     // originial send buffer before load balancing
  void * buffer_out;    // final recv buffer
  void * balance_send_buffer; // buffer for sending load-balanced data
  void * balance_recv_buffer; // buffer for receiving load-balanced data
  void * recv_complete_signal;
  void * redistribute_complete_signal;
  struct flash_shared_memory_t share_mem;
  // NVSHMEM buffers for inter-node transfer
  uint8_t * send_buffer;  // sending buffer after load balancing
  uint8_t * internode_buffer1;  // ping-pong buffer for receiving cross-node traffic
  uint64_t * sync_signal1;      // server_n, since only i-th GPU talks to i-th GPU on each server
  uint8_t * internode_buffer2;  // ping-pong buffer for receiving cross-node traffic
  uint64_t * sync_signal2;
  // NVSHMEM for staging
  uint64_t * credit;
  #if VERIFY_BUFFER == 1
  void * buffer_verify;
#endif
};

struct flash_buffer_ptr_t init_flash_buffer(struct flash_buffer_sz_params_t * params, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n, uint data_size);
void free_flash_buffer(struct flash_buffer_ptr_t * buf);

struct intranode_transfer_params_t init_intranode_transfer_params(struct intra_then_inter_data_buffer_t * data_buf, uint this_rank, uint rank_n_per_node, uint rank_n, struct kernel_sz_t * ksz);
void free_intranode_transfer_params(struct intranode_transfer_params_t * intra_params);

struct internode_transfer_params_t init_internode_transfer_params(struct intra_then_inter_data_buffer_t * data_buf, uint this_rank, uint rank_n_per_node, uint rank_n);
void free_internode_transfer_params(struct internode_transfer_params_t * inter_params);


int launch_fanout_alltoallv(
    struct fanout_data_buffer_t * data_buf,
    uint32_t this_rank,
    uint32_t rank_n_per_node,
    uint32_t rank_n,
    struct fanout_nvshmem_buffer_t * buf,
    struct fanout_buffer_params_t * params,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream
);


int launch_alltoallv(
    struct intra_then_inter_data_buffer_t * data_buf,
    uint32_t this_rank,
    uint32_t rank_n_per_node,
    uint32_t rank_n,
    struct fanout_nvshmem_buffer_t * buf,
    struct intranode_transfer_params_t * intra_params,
    struct internode_transfer_params_t * inter_params,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2
);

#if VERIFY_BUFFER == 1
void verify_intra_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n_per_node, uint rank_n);
void verify_inter_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n_per_node, uint rank_n);
void verify_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n);
#endif

struct fanout_data_buffer_t init_fanout_data_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n);
void free_fanout_data_buffer(struct fanout_data_buffer_t * buf);

struct fanout_nvshmem_buffer_t init_fanout_nvshmem_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n);
void free_fanout_nvshmem(struct fanout_nvshmem_buffer_t * buf);
struct fanout_buffer_params_t init_fanout_params(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, struct kernel_sz_t * ksz);
void free_fanout_params(struct fanout_buffer_params_t * params);

void set_kernel_sharemem_sz( struct kernel_sz_t * ksz);

// FLASH kernel
int launch_flash_alltoallv(
    struct flash_buffer_ptr_t * buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t stream3,
    cudaEvent_t * events,
    uint event_n,
    uint data_size,
    uint credit_flip
);

int launch_flash_alltoallv_chunk(
    struct flash_buffer_ptr_t * buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t stream3,
    cudaEvent_t * events,
    uint event_n,
    uint credit_flip
);


#if VERIFY_BUFFER == 1
void verify_flash_buffers(struct flash_buffer_ptr_t * buf, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n);
void verify_flash_intra_alltoall_buffers(struct flash_buffer_ptr_t * buf, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n);
void verify_flash_inter_alltoall_buffers(struct flash_buffer_ptr_t * buf, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n);
#endif