#pragma once
#include <iostream>
#include "alltoall_matrix.h"
#include "alltoall_local_scheduler.h"
#include "alltoall_algorithm.h"
#include "alltoall_define.h"


struct scheduling_step_t{
    uint to_server[MAX_SERVER_NUM];
    uint from_server[MAX_SERVER_NUM];
    // ChannelPtr: gpu_n * gpu_n (row -> remote dst_gpu's local id, col -> from gpu)
    uint64_t channel[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
    uint64_t crossnode_sz[MAX_SERVER_NUM][MAX_GPU_PER_SERVER];
    uint64_t restore_alltoall_sz[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER];
    //  RestorePtr: gpu_n * gpu_n (row -> dst_gpu's local id, col -> from gpu)
    struct recv_data_t restore[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
    // server id * channel id
    struct recv_data_t direct_cpy[MAX_SERVER_NUM][MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE];
};

struct scheduling_result_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    struct balance_data_t balance[MAX_SERVER_NUM][MAX_SERVER_NUM][MAX_GPU_PER_SERVER_SQUARE];
    struct scheduling_step_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;
    uint64_t intrinsic_ata[MAX_SERVER_NUM][MAX_GPU_PER_SERVER_SQUARE];
};


struct memcpy_buffer_t{
    uint64_t src_disp;
    uint64_t dst_disp;
    uint64_t sz;
};

struct send_recv_buffer_t{
    uint64_t gpu;
    uint64_t disp;
    uint64_t sz;
};

struct scheduling_step_gpu_t{
    struct send_recv_buffer_t crossnode_send;
    struct send_recv_buffer_t crossnode_recv;
    struct send_recv_buffer_t restore_send[GPU_NUM_PER_SERVER];
    uint restore_send_n;
    struct send_recv_buffer_t restore_recv[GPU_NUM_PER_SERVER];
    uint restore_recv_n;
    struct memcpy_buffer_t direct_memcpy[GPU_NUM_PER_SERVER];
    uint direct_memcpy_n;
    struct memcpy_buffer_t restore_memcpy[MAX_SERVER_NUM_SQUARE];
    uint restore_memcpy_n;
};

// scheduling result for a particular gpu
struct scheduling_result_gpu_t{
    uint gpu_n;
    uint server_n;
    uint rankid;
    // intrinsic alltoall metadata
    struct send_recv_buffer_t intrinsic_send[GPU_NUM_PER_SERVER];
    uint intrinsic_send_n;
    struct send_recv_buffer_t intrinsic_recv[GPU_NUM_PER_SERVER];
    uint intrinsic_recv_n;
    // load balance metadata
    struct send_recv_buffer_t balance_send[GPU_NUM_PER_SERVER];
    uint balance_send_n;
    struct send_recv_buffer_t balance_recv[GPU_NUM_PER_SERVER];
    uint balance_recv_n;

    struct memcpy_buffer_t balance_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE];
    uint balance_memcpy_n;

    struct scheduling_step_gpu_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;

#if ABLATION_TEST
    struct send_recv_buffer_t ablation_crossnode_send[MAX_SERVER_NUM];
    uint ablation_crossnode_send_n;
    struct send_recv_buffer_t ablation_crossnode_recv[MAX_SERVER_NUM];
    uint ablation_crossnode_recv_n;
    struct send_recv_buffer_t ablation_restore_send[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint ablation_restore_send_n;
    struct send_recv_buffer_t ablation_restore_recv[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint ablation_restore_recv_n;
    struct memcpy_buffer_t ablation_direct_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint ablation_direct_memcpy_n;
    struct memcpy_buffer_t ablation_restore_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint ablation_restore_memcpy_n;
#endif
};

struct chunk_metadata_t{
    uint64_t chunk_ready_threshold[FLASH_MAX_CHUNK_NUM];    // we don't want too many chunks
    uint64_t chunk_sz[FLASH_MAX_CHUNK_NUM];
    uint64_t chunk_disp[FLASH_MAX_CHUNK_NUM];
    uint chunk_n;
};

struct device_inter_p2p_params_t{
    // for sending
    uint dst_rank;
    uint64_t send_sz;
    uint64_t src_disp;
    // for recving
    uint src_rank;
    uint64_t dst_disp;
    uint64_t recv_sz;
};

struct device_intra_alltoall_params_t{
    uint64_t sz[GPU_NUM_PER_SERVER_TIMES_MAX_BLOCK_NUM]; // transfer size for each sender, local_rank_n * block_n
    uint64_t src_disp[GPU_NUM_PER_SERVER_TIMES_MAX_BLOCK_NUM];  // send disp for each sender,  local_rank_n * block_n
    uint64_t dst_disp[GPU_NUM_PER_SERVER_TIMES_MAX_BLOCK_NUM];  // recv disp for each sender, local_rank_n * block_n
    uint64_t round_n[GPU_NUM_PER_SERVER_TIMES_MAX_BLOCK_NUM];      // round n for each sender, local_rank_n * block_n
    uint64_t recv_sz[GPU_NUM_PER_SERVER];
};

struct device_intra_redistribute_alltoall_params_t{
    uint64_t sz[GPU_NUM_PER_SERVER_SQUARE_TIMES_MAX_BLOCK_NUM]; // transfer size for each sender, local_rank_n (src local gpu id) * local_rank_n (dst local gpu id) * block_n
    uint64_t src_disp[GPU_NUM_PER_SERVER_SQUARE_TIMES_MAX_BLOCK_NUM];  // send disp for each sender,  local_rank_n (src local gpu id) * local_rank_n (dst local gpu id) * block_n
    uint64_t dst_disp[GPU_NUM_PER_SERVER_SQUARE_TIMES_MAX_BLOCK_NUM];  // recv disp for each sender, local_rank_n (src local gpu id) * local_rank_n (dst local gpu id) * block_n
    uint64_t round_n[GPU_NUM_PER_SERVER_SQUARE_TIMES_MAX_BLOCK_NUM];      // round n for each sender, local_rank_n (src local gpu id) * local_rank_n (dst local gpu id) * block_n
    uint64_t recv_sz[GPU_NUM_PER_SERVER];
};

void plan_chunk_pipeline(struct chunk_metadata_t * param, uint64_t p2p_sz);
void assign_intra_alltoall_to_blocks(struct device_intra_alltoall_params_t * param, uint64_t * src_disp, uint64_t * dst_disp, uint64_t * sz, uint local_rank_n, uint block_n, uint64_t block_tx_burst);
void assign_intra_redistribute_to_blocks(struct device_intra_redistribute_alltoall_params_t * param, uint64_t * src_disp, uint64_t * dst_disp, uint64_t * sz, uint local_rank_n, uint block_n, uint64_t block_tx_burst);

struct flash_schedule_metadata_t{
    uint this_rank;
    uint local_rank_n;
    uint rank_n;
    uint server_id;
    uint server_n;
    uint block_n;
    uint64_t min_tx_sz_at_a_time;       // minimum transfer size at a time
};

struct flash_schedule_this_gpu_t{
    struct flash_schedule_metadata_t info;
    // metadata for intrinsic alltoall
    struct device_intra_alltoall_params_t * dev_intrinsic_alltoall;
    struct device_intra_alltoall_params_t * host_intrinsic_alltoall;
    // metadata for load balance
    struct device_intra_alltoall_params_t * dev_balance_alltoall;
    struct device_intra_alltoall_params_t * host_balance_alltoall;
    struct memcpy_buffer_t balance_memcpy[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE];
    uint balance_memcpy_n;
    // metadata for sync threshold
    uint64_t * host_sync_threshold;
    // // memory for grid synchronization
    // unsigned * bar_count;
    // unsigned * bar_sense;
    // metadata for pipeline stages
    uint stage_n;
    struct device_inter_p2p_params_t * dev_stages_internode;    //MAX_TRANSFER_STEP_NUM
    struct device_inter_p2p_params_t * host_stages_internode;    //MAX_TRANSFER_STEP_NUM
    struct device_intra_redistribute_alltoall_params_t * dev_stages_intra_redistribute;  //MAX_TRANSFER_STEP_NUM
    struct device_intra_redistribute_alltoall_params_t * host_stages_intra_redistribute;  //MAX_TRANSFER_STEP_NUM
    struct chunk_metadata_t * dev_stages_send_chunks;
    struct chunk_metadata_t * host_stages_send_chunks;
    struct chunk_metadata_t * dev_stages_recv_chunks;
    struct chunk_metadata_t * host_stages_recv_chunks;
};

struct sendbuff_region_t{
    uint64_t src_gpu_disp[GPU_NUM_PER_SERVER];
    uint64_t src_gpu_sz[GPU_NUM_PER_SERVER];
    uint64_t src_gpu_offset[GPU_NUM_PER_SERVER];
    uint src_gpu_n;
};


struct lbbuff_region_t{
    uint64_t server_disp[MAX_SERVER_NUM];
    uint64_t server_sz[MAX_SERVER_NUM];
    uint64_t server_offset[MAX_SERVER_NUM];
    uint server_n;
};

struct lbbuff_area_t{
    struct lbbuff_region_t dst_gpu_region[GPU_NUM_PER_SERVER];
    uint64_t dst_gpu_disp[GPU_NUM_PER_SERVER];
    uint64_t dst_gpu_sz[GPU_NUM_PER_SERVER];
    uint dst_gpu_n;
};

struct buffer_parameter_t{
    uint64_t inputbuff_disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t inputbuff_sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t inputbuff_total_sz;

    uint64_t sendbuff_disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t sendbuff_sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    struct sendbuff_region_t sendbuff_region[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t sendbuff_total_sz;

    uint64_t recvbuff_disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t recvbuff_sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];
    uint64_t recvbuff_total_sz;


    uint64_t lbsend_disp[GPU_NUM_PER_SERVER];
    uint64_t lbsend_sz[GPU_NUM_PER_SERVER];
    struct lbbuff_area_t lbsend_area[GPU_NUM_PER_SERVER];
    uint64_t lbsend_total_sz;

    uint64_t lbrecv_disp[GPU_NUM_PER_SERVER];
    uint64_t lbrecv_sz[GPU_NUM_PER_SERVER];
    struct lbbuff_area_t lbrecv_area[GPU_NUM_PER_SERVER];
    uint64_t lbrecv_total_sz;

    uint64_t crosbuff_total_sz; // use the offset to alternate the first and second half of the buffer
    uint64_t crosbuff_offset;
    uint64_t rstrbuff_total_sz;

#if ABLATION_TEST
    uint64_t ablation_crosbuff_total_sz;
    uint64_t ablation_rstrbuff_total_sz;
#endif

};

struct flash_balance_per_local_rank_param_t{
    uint64_t disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // rank_n, actual dst/src rank
    uint64_t sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];   // rank_n, actual dst/src rank
    uint64_t rank_n;
    uint64_t data_offset[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // local disp at memory 
};

struct flash_balance_param_t{
    uint64_t total_sz; // total size of the balanced data to be sent
    uint64_t disp[GPU_NUM_PER_SERVER];  // local_rank_n, src/dst local rank
    uint64_t sz[GPU_NUM_PER_SERVER];    // local_rank_n, src/dst local rank
    struct flash_balance_per_local_rank_param_t layout[GPU_NUM_PER_SERVER]; 
};

struct flash_send_per_dst_rank_param_t{
    uint64_t src_gpu_disp[GPU_NUM_PER_SERVER];
    uint64_t src_gpu_sz[GPU_NUM_PER_SERVER];
    uint64_t data_offset[GPU_NUM_PER_SERVER]; // where to copy data from in the input buffer
    uint src_gpu_n;
};

struct flash_send_param_t{
    uint64_t total_sz; // total size of the data to be sent
    uint64_t max_total_sz;  // the size of buffer to be allocated
    uint64_t disp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // rank_n,
    uint64_t sz[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];   // rank_n,
    struct flash_send_per_dst_rank_param_t layout[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; // rank_n, actual dst rank
};

struct flash_buffer_sz_params_t{
    // input and output buffer
    uint64_t in_disp_per_rank[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];    // rank_n, for each dst rank, the disp in the current rank's buffer in
    uint64_t in_sz_per_rank[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER]; 
    uint64_t in_total_sz;
    uint64_t out_disp_per_rank[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];   // rank_n, for each src rank, the disp in the current rank's buffer out
    uint64_t out_sz_per_rank[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER];  
    uint64_t out_total_sz;
    // load-balance buffer
    struct flash_balance_param_t balance_send_param; // load-balance send buffer
    struct flash_balance_param_t balance_recv_param; // load-balance recv buffer
    // send buffer (NVSHMEM)
    struct flash_send_param_t send_param; // send buffer
    // receive pingpong buffer (NVSHMEM)
    uint64_t pingpong_sz; // total size of ping-pong receive buffer
};

struct per_step_profile_t{
    double step_time;
    double cross_node;
    double intra_all2all;
};

struct profiler_t{
    uint64_t workload_sz;   // in unit of element#
    // calculate the transfer completion time and tput limit: suppose intra-host network is infinitely fast: load balance and data redistribution are instantly completed
    // the transfer completion time is basically the time needed for transfering over inter-server network
    // this does not include memory copy time
    double crossnode_time_limit;
    double algbw_limit;     
    // estimate FLASH completion time under current schedule
    double flash_total_time;
    double load_balance;
    double load_balance_per_server[MAX_SERVER_NUM];
    struct per_step_profile_t steps[MAX_TRANSFER_STEP_NUM];
    uint step_n;
};

struct GlobalScheduler{
    uint server_n;
    uint gpu_n;
    uint block_n;
    uint64_t * temp_data;
    struct Matrix mat;
    struct LocalScheduler * locals[MAX_SERVER_NUM];
    struct scheduling_result_t * sched;
    struct scheduling_result_gpu_t * gpu_sched;
    struct flash_schedule_this_gpu_t * flash_sched;
    struct buffer_parameter_t * buff_parameter;
    struct flash_buffer_sz_params_t * flash_buffer_sz_params;
    struct profiler_t opt;
    uint data_size;
};


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix, uint rankid, uint block_n, uint64_t block_tx_burst, uint data_sz);
void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix, uint rankid);
void free_global_scheduler(struct GlobalScheduler * gs);
void update_global_scheduler(struct GlobalScheduler * gs, uint64_t * demand_matrix);
void run_scheduler(struct GlobalScheduler * gs);
void flash_scheduler(struct GlobalScheduler * gs);
void get_buffer_size(struct GlobalScheduler * gs);
void get_flash_buffer_size(struct GlobalScheduler * gs);
void schedule_this_gpu(struct GlobalScheduler * gs);
void flash_schedule_this_gpu(struct GlobalScheduler * gs);
double alltoall_completion_time_with_spreadout(uint64_t * workload, uint dim, double tput);

#if ABLATION_TEST
void run_scheduler_ablation(struct GlobalScheduler * gs);
void schedule_this_gpu_ablation(struct GlobalScheduler * gs);
#endif



