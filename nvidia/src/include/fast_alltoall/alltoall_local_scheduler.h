#pragma once

#include "alltoall_define.h"

struct data_t{
    uint64_t sz[MAX_GPU_PER_SERVER];        // actual source gpu
    uint64_t offset[MAX_GPU_PER_SERVER];
    uint64_t sum;
};

struct recv_data_t{
    uint64_t sz;
    uint64_t offset;
};

struct balance_data_t{
    uint64_t sz[MAX_GPU_PER_SERVER];    // transferred data size for each local gpu
    uint64_t offset[MAX_GPU_PER_SERVER];
};

struct LocalScheduler{
    uint64_t ** data;
    uint64_t ** balanced_data;
    struct data_t ** data_after_balance;
    uint64_t * intrinsic_all2all;
    uint gpu_n;
    uint server_n;
    uint server_id;
    uint64_t * row_sum; //row sum at each tile; gpu_n * server_n
    uint64_t * server2server_data;
};

void init_local_scheduler(struct LocalScheduler * ls, uint64_t* _data, uint _gpu_n, uint _server_n, uint _server_id);
void update_local_scheduler(struct LocalScheduler * ls, uint64_t* _data);
void free_local_scheduler(struct LocalScheduler * ls);
void prepare_load_balance(struct LocalScheduler * ls);
void balance_one_server(struct LocalScheduler * ls, uint to_server_id, struct balance_data_t (*r)[MAX_GPU_PER_SERVER_SQUARE]);   // r is a transfer matrix before this server talks to another server to balance data
void restore_one_server(struct LocalScheduler * ls, uint to_server_id, uint64_t (*channel)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t (*crossnode_sz)[MAX_GPU_PER_SERVER], struct recv_data_t (*r)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t (*restore_alltoall_sz)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER], struct recv_data_t (*dcpy)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t freq);    // r is a transfer matrix after this server talks to another server to restore data
void print_local_scheduler(struct LocalScheduler * ls);
void print_local_scheduler(struct LocalScheduler * ls, uint dst_server_id);

void print_matrix(uint64_t * data, uint m, uint n);