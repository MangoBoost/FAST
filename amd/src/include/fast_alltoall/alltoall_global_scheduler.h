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

struct theoretical_optimality_t{
    double algbw_limit;
    double crossnode_time_limit;
};

struct GlobalScheduler{
    uint server_n;
    uint gpu_n;
    uint64_t * temp_data;
    struct Matrix mat;
    struct LocalScheduler * locals[MAX_SERVER_NUM];
    struct scheduling_result_t * sched;
    struct scheduling_result_gpu_t * gpu_sched;
    struct buffer_parameter_t * buff_parameter;
    struct theoretical_optimality_t opt;
};


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix, uint rankid);
void free_global_scheduler(struct GlobalScheduler * gs);
void update_global_scheduler(struct GlobalScheduler * gs, uint64_t * demand_matrix);
void run_scheduler(struct GlobalScheduler * gs);
void get_buffer_size(struct GlobalScheduler * gs);
void schedule_this_gpu(struct GlobalScheduler * gs);

#if ABLATION_TEST
void run_scheduler_ablation(struct GlobalScheduler * gs);
void schedule_this_gpu_ablation(struct GlobalScheduler * gs);
#endif



