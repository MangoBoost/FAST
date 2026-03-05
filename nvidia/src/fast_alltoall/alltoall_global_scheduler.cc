#include "fast_alltoall/alltoall_global_scheduler.h"
#include "fast_alltoall/alltoall_define.h"
#include <chrono>
#include <iostream>
#if ROCM_RCCL_COMPILE
#include <hip/hip_runtime.h>
#endif
#if CUDA_NCCL_COMPILE
#include <cuda_runtime.h>
#endif

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix, uint rankid){
    gs->server_n = _server_n;
    gs->gpu_n = _gpu_n;
    uint dim = gs->gpu_n * gs->server_n;
    for (uint s = 0; s < gs->server_n; s++){
        gs->locals[s] = (LocalScheduler *) malloc(sizeof(LocalScheduler));
        // hipMallocManaged((void**)&gs->locals[s], sizeof(LocalScheduler));
        init_local_scheduler(gs->locals[s], demand_matrix + s * dim * gs->gpu_n, gs->gpu_n, gs->server_n, s);
    }
    gs->temp_data = (uint64_t *) malloc(sizeof(uint64_t) * gs->server_n * gs->server_n);
    // hipMallocManaged((void**)&data, sizeof(uint) * gs->server_n * gs->server_n);
    // uint *data = new uint[server_n * server_n];
    for (uint s = 0; s < gs->server_n; s++){
       uint src_svr = gs->locals[s] -> server_id;
        for (uint j = 0; j < gs->server_n; j++){
            gs->temp_data[src_svr * gs->server_n + j] =  gs->locals[s]->server2server_data[j];
       }
    }
    init_matrix(&gs->mat);
    copy_matrix(&gs->mat, gs->temp_data, gs->server_n);
    gs->sched = (scheduling_result_t *) malloc(sizeof(scheduling_result_t));
    memset(gs->sched, 0, sizeof(scheduling_result_t));
    // hipMallocManaged((void**) &gs->sched, sizeof(scheduling_result_t));
    // hipMemset(gs->sched, 0, sizeof(scheduling_result_t));
    gs->sched->gpu_n = _gpu_n;
    gs->sched->server_n = _server_n;
    gs->sched->rankid = rankid;

    gs->gpu_sched = (scheduling_result_gpu_t *) malloc(sizeof(scheduling_result_gpu_t));
    memset(gs->gpu_sched, 0, sizeof(scheduling_result_gpu_t));
    gs->gpu_sched->gpu_n = _gpu_n;
    gs->gpu_sched->server_n = _server_n;
    gs->gpu_sched->rankid = rankid;

    gs -> buff_parameter = (buffer_parameter_t *) malloc(sizeof(buffer_parameter_t));
    memset(gs -> buff_parameter, 0 , sizeof(buffer_parameter_t));

    gs -> opt.algbw_limit = 0;
    gs -> opt.crossnode_time_limit = 0;
}

void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix,  uint rankid, uint _block_n, uint64_t block_tx_burst, uint data_sz){
    gs->server_n = _server_n;
    gs->gpu_n = _gpu_n;
    gs->block_n = _block_n;
    gs ->data_size = data_sz;
    uint dim = gs->gpu_n * gs->server_n;
    for (uint s = 0; s < gs->server_n; s++){
        gs->locals[s] = (LocalScheduler *) malloc(sizeof(LocalScheduler));
        // hipMallocManaged((void**)&gs->locals[s], sizeof(LocalScheduler));
        init_local_scheduler(gs->locals[s], demand_matrix + s * dim * gs->gpu_n, gs->gpu_n, gs->server_n, s);
    }
    gs->temp_data = (uint64_t *) malloc(sizeof(uint64_t) * gs->server_n * gs->server_n);
    // hipMallocManaged((void**)&data, sizeof(uint) * gs->server_n * gs->server_n);
    // uint *data = new uint[server_n * server_n];
    for (uint s = 0; s < gs->server_n; s++){
       uint src_svr = gs->locals[s] -> server_id;
        for (uint j = 0; j < gs->server_n; j++){
            gs->temp_data[src_svr * gs->server_n + j] =  gs->locals[s]->server2server_data[j];
       }
    }
    init_matrix(&gs->mat);
    copy_matrix(&gs->mat, gs->temp_data, gs->server_n);
    gs->sched = (scheduling_result_t *) malloc(sizeof(scheduling_result_t));
    memset(gs->sched, 0, sizeof(scheduling_result_t));
    // hipMallocManaged((void**) &gs->sched, sizeof(scheduling_result_t));
    // hipMemset(gs->sched, 0, sizeof(scheduling_result_t));
    gs->sched->gpu_n = _gpu_n;
    gs->sched->server_n = _server_n;
    gs->sched->rankid = rankid;

    gs->gpu_sched = (scheduling_result_gpu_t *) malloc(sizeof(scheduling_result_gpu_t));
    memset(gs->gpu_sched, 0, sizeof(scheduling_result_gpu_t));
    gs->gpu_sched->gpu_n = _gpu_n;
    gs->gpu_sched->server_n = _server_n;
    gs->gpu_sched->rankid = rankid;

    gs -> buff_parameter = (buffer_parameter_t *) malloc(sizeof(buffer_parameter_t));
    memset(gs -> buff_parameter, 0 , sizeof(buffer_parameter_t));

    gs -> flash_buffer_sz_params = (flash_buffer_sz_params_t *) malloc(sizeof(flash_buffer_sz_params_t));
    memset(gs -> flash_buffer_sz_params, 0 , sizeof(flash_buffer_sz_params_t));
    memset(&(gs -> opt), 0, sizeof(struct profiler_t));

    // init flash sched
    gs->flash_sched = (struct flash_schedule_this_gpu_t *) malloc(sizeof(struct flash_schedule_this_gpu_t));
    memset(&gs->flash_sched->info , 0, sizeof(struct flash_schedule_metadata_t));
    gs->flash_sched->info.this_rank = rankid;
    gs->flash_sched->info.local_rank_n = _gpu_n;
    gs->flash_sched->info.rank_n = _server_n * _gpu_n;
    gs->flash_sched->info.server_id = rankid / _gpu_n;
    gs->flash_sched->info.server_n = _server_n;
    gs->flash_sched->info.block_n = _block_n;
    gs->flash_sched->info.min_tx_sz_at_a_time = block_tx_burst;
    // init memory for scheduling result
    // intrinsic alltoall
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_intrinsic_alltoall, sizeof(struct device_intra_alltoall_params_t)));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_intrinsic_alltoall, 0, sizeof(struct device_intra_alltoall_params_t)));
    gs->flash_sched->host_intrinsic_alltoall = (struct device_intra_alltoall_params_t *)malloc(sizeof(struct device_intra_alltoall_params_t));
    memset(gs->flash_sched->host_intrinsic_alltoall, 0 , sizeof(struct device_intra_alltoall_params_t));
    // load balance
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_balance_alltoall, sizeof(struct device_intra_alltoall_params_t)));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_balance_alltoall, 0, sizeof(struct device_intra_alltoall_params_t)));
    gs->flash_sched->host_balance_alltoall = (struct device_intra_alltoall_params_t *)malloc(sizeof(struct device_intra_alltoall_params_t));
    memset(gs->flash_sched->host_balance_alltoall, 0, sizeof(struct device_intra_alltoall_params_t));
    memset(gs->flash_sched->balance_memcpy, 0, sizeof(struct memcpy_buffer_t ) * MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE);
    gs->flash_sched->balance_memcpy_n = 0;
    // pipeline stages
    gs->flash_sched->stage_n = 0;
    // inter-node metadata
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_stages_internode, sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_internode, 0, sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM));
    gs->flash_sched->host_stages_internode = (struct device_inter_p2p_params_t *)malloc(sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_stages_internode, 0, sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM);
    // intra-node redistribution metadata
    // metadata for grid sync
    // CUDACHECK(cudaMalloc((void**)&gs->flash_sched->bar_count, sizeof(unsigned) * MAX_TRANSFER_STEP_NUM));
    // CUDACHECK(cudaMemset(gs->flash_sched->bar_count, 0, sizeof(unsigned) * MAX_TRANSFER_STEP_NUM));
    // CUDACHECK(cudaMalloc((void**)&gs->flash_sched->bar_sense, sizeof(unsigned) * MAX_TRANSFER_STEP_NUM));
    // CUDACHECK(cudaMemset(gs->flash_sched->bar_sense, 0, sizeof(unsigned)* MAX_TRANSFER_STEP_NUM));
    gs->flash_sched->host_sync_threshold = (uint64_t *) malloc(sizeof(uint64_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_sync_threshold, 0, sizeof(uint64_t) * MAX_TRANSFER_STEP_NUM);
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_stages_intra_redistribute, sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_intra_redistribute, 0, sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM));
    gs->flash_sched->host_stages_intra_redistribute = (struct device_intra_redistribute_alltoall_params_t *)malloc(sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_stages_intra_redistribute, 0, sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM);
    // chunk metadata
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_stages_send_chunks, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_send_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMalloc((void**)&gs->flash_sched->dev_stages_recv_chunks, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_recv_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    gs->flash_sched->host_stages_send_chunks = (struct chunk_metadata_t *) malloc(sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_stages_send_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
    gs->flash_sched->host_stages_recv_chunks = (struct chunk_metadata_t *) malloc(sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_stages_recv_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
    CUDACHECK(cudaDeviceSynchronize());
}

void update_global_scheduler(struct GlobalScheduler * gs, uint64_t * demand_matrix){
    uint dim = gs->gpu_n * gs->server_n;
    for (uint s = 0; s < gs->server_n; s++){
        update_local_scheduler(gs->locals[s], demand_matrix + s * dim * gs->gpu_n);
    }
    for (uint s = 0; s < gs->server_n; s++){
        uint src_svr = gs->locals[s] -> server_id;
        for (uint j = 0; j < gs->server_n; j++){
            gs->temp_data[src_svr * gs->server_n + j] =  gs->locals[s]->server2server_data[j];
       }
    }
    update_matrix(&gs->mat);
    copy_matrix(&gs->mat, gs->temp_data, gs->server_n);

    uint rankid = gs->gpu_sched->rankid;
    memset(gs->sched, 0, sizeof(scheduling_result_t));
    gs->sched->gpu_n = gs->gpu_n;
    gs->sched->server_n = gs->server_n;

    memset(gs->gpu_sched, 0, sizeof(scheduling_result_gpu_t));
    gs->gpu_sched->gpu_n = gs->gpu_n;
    gs->gpu_sched->server_n = gs->server_n;
    gs->gpu_sched->rankid = rankid;
    gs->gpu_sched->step_n = 0;

    memset(gs -> buff_parameter, 0 , sizeof(buffer_parameter_t));
    memset(gs -> flash_buffer_sz_params, 0 , sizeof(flash_buffer_sz_params_t));

    gs -> opt.algbw_limit = 0;
    gs -> opt.crossnode_time_limit = 0;

    // CUDACHECK(cudaMemset(gs->flash_sched->bar_count, 0, sizeof(unsigned) * MAX_TRANSFER_STEP_NUM));
    // CUDACHECK(cudaMemset(gs->flash_sched->bar_sense, 0, sizeof(unsigned) * MAX_TRANSFER_STEP_NUM));

    CUDACHECK(cudaMemset(gs->flash_sched->dev_intrinsic_alltoall, 0, sizeof(struct device_intra_alltoall_params_t)));
    memset(gs->flash_sched->host_intrinsic_alltoall, 0, sizeof(struct device_intra_alltoall_params_t));

    CUDACHECK(cudaMemset(gs->flash_sched->dev_balance_alltoall, 0, sizeof(struct device_intra_alltoall_params_t)));
    memset(gs->flash_sched->host_balance_alltoall, 0, sizeof(struct device_intra_alltoall_params_t));
    memset(gs->flash_sched->balance_memcpy, 0, sizeof(struct memcpy_buffer_t ) * MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER_SQUARE);
    gs->flash_sched->balance_memcpy_n = 0;
    gs->flash_sched->stage_n = 0;

    memset(gs->flash_sched->host_sync_threshold, 0, sizeof(uint64_t) * MAX_TRANSFER_STEP_NUM);
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_internode, 0, sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM));
    memset(gs->flash_sched->host_stages_internode, 0, sizeof(struct device_inter_p2p_params_t) * MAX_TRANSFER_STEP_NUM);
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_intra_redistribute, 0, sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM));
    memset(gs->flash_sched->host_stages_intra_redistribute, 0, sizeof(struct device_intra_redistribute_alltoall_params_t) * MAX_TRANSFER_STEP_NUM);

    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_send_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    CUDACHECK(cudaMemset(gs->flash_sched->dev_stages_recv_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM));
    memset(gs->flash_sched->host_stages_send_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
    memset(gs->flash_sched->host_stages_recv_chunks, 0, sizeof(struct chunk_metadata_t) * MAX_TRANSFER_STEP_NUM);
}


void free_global_scheduler(struct GlobalScheduler * gs){
    free_matrix(&gs->mat);
     for (uint s = 0; s < gs->server_n; s++){
        free_local_scheduler(gs->locals[s]);
    }
    free(gs->temp_data);
    free(gs->sched);
    free(gs->gpu_sched);
    free(gs->buff_parameter);
    free(gs->flash_buffer_sz_params);
    // CUDACHECK(cudaFree(gs->flash_sched->bar_count));
    // CUDACHECK(cudaFree(gs->flash_sched->bar_sense));
    CUDACHECK(cudaFree(gs->flash_sched->dev_intrinsic_alltoall));
    free(gs->flash_sched->host_intrinsic_alltoall);
    CUDACHECK(cudaFree(gs->flash_sched->dev_balance_alltoall));
    free(gs->flash_sched->host_balance_alltoall);
    CUDACHECK(cudaFree(gs->flash_sched->dev_stages_internode));
    free(gs->flash_sched->host_stages_internode);
    free(gs->flash_sched->host_sync_threshold);
    CUDACHECK(cudaFree(gs->flash_sched->dev_stages_intra_redistribute));
    free(gs->flash_sched->host_stages_intra_redistribute);
    CUDACHECK(cudaFree(gs->flash_sched->dev_stages_send_chunks));
    CUDACHECK(cudaFree(gs->flash_sched->dev_stages_recv_chunks));
    free(gs->flash_sched->host_stages_send_chunks);
    free(gs->flash_sched->host_stages_recv_chunks);
    free(gs->flash_sched);
}

void flash_scheduler(struct GlobalScheduler * gs){
    FastAll2All all2all;
    init_fastall2all(&all2all, &gs->mat);
    to_scaled_doubly_stochastic_matrix_fastall2all(&all2all);
    decompose_fastall2all(&all2all);
    // LOG("verify deccomposition: %u\n", verify_decomposition_fastall2all(&all2all));
    // if (gs -> sched->rankid ==0){
    //     std::cout << "[PSETS]: ";
    //     for (uint i = 0; i < all2all.p_sets_n; i++){
    //         std::cout << get_freq_permutation_set(&all2all.p_sets[all2all.p_sets_ascending[i]]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    uint pid = 0, lid = 0;

    /* Start Pipelining*/

    // generate schedule for intra-server all2all - balance first
    // balance once

    for (lid = 0; lid < gs->server_n; lid++){
        for (uint s = 0; s < gs->server_n; s++){
           if (s == gs->locals[lid]->server_id){
            continue;
           }
           uint src_svr = gs->locals[lid]->server_id;
           balance_one_server(gs->locals[lid], s, &gs->sched->balance[src_svr][s]);
        }
    }

    get_flash_buffer_size(gs);
    get_buffer_size(gs);

    // get intrinsic all-to-all
    for (lid = 0; lid < gs->server_n; lid++){
        uint src_svr = gs->locals[lid]->server_id;
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint64_t));
        // hipMemcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(TransferMatrixElement), hipMemcpyHostToHost);
    }

    uint step_id = 0;
    uint stage_order[all2all.p_sets_n];
    // stage_order[0] = all2all.p_sets_ascending[all2all.p_sets_n - 1]; // put the largest cross-node transfer at first to pipeline with intrinsic alltoall
    for (int i = 0; i < all2all.p_sets_n; i++){
        stage_order[i] = all2all.p_sets_ascending[all2all.p_sets_n - 1 - i];
        // stage_order[i] = all2all.p_sets_ascending[i];
    }
    
    for (int i = 0; i < all2all.p_sets_n; i++){
        pid = stage_order[i];

        for (lid = 0; lid < gs->server_n; lid++){
            uint src_svr = gs->locals[lid]->server_id;
            uint dst_svr = 0;
            map_lookup(all2all.p_sets[pid].mp, all2all.p_sets[pid].mp_n, src_svr, &dst_svr);
            restore_one_server(gs->locals[lid],
                            dst_svr, &gs->sched->steps[step_id].channel[src_svr],
                            &gs->sched->steps[step_id].crossnode_sz[src_svr],
                            &gs->sched->steps[step_id + 1].restore[src_svr],
                            &gs->sched->steps[step_id + 1].restore_alltoall_sz[src_svr],
                            &gs->sched->steps[step_id + 1].direct_cpy[src_svr],
                            get_freq_permutation_set(&all2all.p_sets[pid]));
        }
        to_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].to_server);
        from_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].from_server);
        step_id ++;
    }
    gs->sched->step_n = all2all.p_sets_n + 1;

    flash_schedule_this_gpu(gs);
    schedule_this_gpu(gs);

    uint data_size = gs->data_size;
    uint64_t workload_sz = 0;
    uint64_t max_cross_node_send_sz = 0, max_cross_node_recv_sz = 0;
    for (uint i = 0; i < gs->server_n; i++){
        uint64_t cross_node_send_sz = 0, cross_node_recv_sz = 0;
        for (uint j = 0; j < gs->server_n; j++){
            if (i == j) continue;
            cross_node_send_sz += gs->locals[i]->server2server_data[j];
            cross_node_recv_sz += gs->locals[j]->server2server_data[i];
        }
        max_cross_node_send_sz = MAX(cross_node_send_sz, max_cross_node_send_sz);
        max_cross_node_recv_sz = MAX(cross_node_recv_sz, max_cross_node_recv_sz);

        for (uint k = 0; k < gs->gpu_n; k++){
            for (uint z = 0; z <  gs->gpu_n * gs->server_n; z++){
                workload_sz += gs->locals[i]->data[k][z];
            }
        }
    }
    gs -> opt.workload_sz = workload_sz;
    gs -> opt.crossnode_time_limit = MAX(max_cross_node_send_sz, max_cross_node_recv_sz)  * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT; // ms
    gs -> opt.algbw_limit = workload_sz * data_size * 1e-6 / (gs->gpu_n * gs->server_n) / gs -> opt.crossnode_time_limit; //GBps

#if FLASH_PROFILE
    if (gs->gpu_sched->rankid == 0){
        gs -> opt.step_n = gs->sched->step_n;
        uint64_t * traffic_matrix = new uint64_t[gs -> gpu_n * gs -> gpu_n];
        gs -> opt.flash_total_time = 0;
        // load balance    
        gs -> opt.load_balance = 0;
        for (uint server_id = 0; server_id < gs -> server_n; server_id ++){
            memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
            for (uint i = 0; i < gs -> gpu_n; i ++){
                for (uint j = 0; j < gs -> gpu_n; j++){
                    if (i == j) continue;

                    for (uint dst_gpu = 0; dst_gpu < gs -> gpu_n; dst_gpu ++){
                        for (uint s = 0; s <  gs -> server_n; s++){
                            if (server_id == s)
                                continue;
                            traffic_matrix[i * gs -> gpu_n + j] += (gs -> sched -> balance)[server_id][s][i * gs -> gpu_n + j].sz[dst_gpu] * data_size;
                        }
                    }
                }
            }
            gs -> opt.load_balance_per_server[server_id] = alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6); //ms
            gs -> opt.load_balance = MAX(gs -> opt.load_balance, gs -> opt.load_balance_per_server[server_id]);
        }
        gs -> opt.flash_total_time += gs -> opt.load_balance;
        // first step and intrinsic alltoall
        // cross node
        uint64_t cross_node_max = 0;
        for (uint server_id = 0; server_id < gs->server_n; server_id ++){
            for (uint gpu_id = 0; gpu_id < gs->gpu_n; gpu_id++){
                cross_node_max = MAX(cross_node_max, (gs -> sched -> steps)[0].crossnode_sz[server_id][gpu_id]);
            }
        }
        gs -> opt.steps[0].cross_node = cross_node_max * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT;
        // intrinsic alltoall
        double intra_all2all_max_time = 0;
        for (uint server_id = 0; server_id < gs->server_n; server_id ++){
            memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
            for (uint i = 0; i < gs -> gpu_n; i ++){
                for (uint j = 0; j < gs -> gpu_n; j++){
                    if (i == j) continue;
                    traffic_matrix[i * gs -> gpu_n + j] = gs->locals[server_id]->intrinsic_all2all[i * gs -> gpu_n + j] * data_size;
                }
            }
            intra_all2all_max_time = MAX(intra_all2all_max_time, alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6));
        }
        gs -> opt.steps[0].intra_all2all = intra_all2all_max_time;
        gs -> opt.steps[0].step_time = MAX(gs -> opt.steps[0].intra_all2all, gs -> opt.steps[0].cross_node);
        gs -> opt.flash_total_time += gs -> opt.steps[0].step_time;

        // middle steps
        for (uint step_id = 1; step_id <= gs->sched->step_n - 1; step_id ++){
            gs -> opt.steps[step_id].cross_node = 0;
            if (step_id != gs->sched->step_n - 1){
                // cross_node
                cross_node_max = 0;
                for (uint server_id = 0; server_id < gs->server_n; server_id ++){
                    for (uint gpu_id = 0; gpu_id < gs->gpu_n; gpu_id++){
                        cross_node_max = MAX(cross_node_max, (gs -> sched -> steps)[step_id].crossnode_sz[server_id][gpu_id]);
                    }
                }
                gs -> opt.steps[step_id].cross_node = cross_node_max * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT;
            }
            // intra alltoall for data redistribution
            intra_all2all_max_time = 0;
            for (uint server_id = 0; server_id < gs->server_n; server_id ++){
                memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
                uint64_t redis_sum = 0;
                for (uint i = 0; i < gs -> gpu_n; i ++){
                    for (uint j = 0; j < gs -> gpu_n; j++){
                        if (i == j) continue;
                        traffic_matrix[i * gs -> gpu_n + j] =  (gs -> sched -> steps)[step_id].restore_alltoall_sz[server_id][i][j] * data_size;
                        redis_sum += traffic_matrix[i * gs -> gpu_n + j];
                    }
                }
                intra_all2all_max_time = MAX(intra_all2all_max_time, alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6));
                printf("server %u, step %u, redis matrix total sz %lu B\n", server_id, step_id, redis_sum);
            }
            gs -> opt.steps[step_id].intra_all2all = intra_all2all_max_time;
            gs -> opt.steps[step_id].step_time = MAX(gs -> opt.steps[step_id].intra_all2all, gs -> opt.steps[step_id].cross_node);
            gs -> opt.flash_total_time += gs -> opt.steps[step_id].step_time;
        }
        delete[] traffic_matrix;
        // -------------------------------------------------------------------------------
        // PRINT OUT PROFILING RESULT
        //--------------------------------------------------------------------------------
        std::cout << "==================FLASH THEORETICAL RESULT==============================" <<std::endl;
        std::cout << "Achievable throughput: " << gs -> opt.workload_sz * data_size * 1e-6 / (gs->gpu_n * gs->server_n) / gs -> opt.flash_total_time << " GBps" << std::endl
            << "Total completion time : " <<  gs -> opt.flash_total_time << " ms" << std::endl
            << "Load Balance: " << gs -> opt.load_balance << " ms" << std::endl
            << "Step n: " << gs->opt.step_n << std::endl
            << "Optimal cross node time: " << gs->opt.crossnode_time_limit <<  " ms" <<std::endl
            << "Throughput limit: " << gs->opt.algbw_limit << " GBps" << std::endl;

        std::cout << "----------------------------------------------------------------" <<std::endl;

        for (uint i = 0; i < gs->opt.step_n; i++){
            std::cout << "[Step " << i << "] Time: " <<  gs -> opt.steps[i].step_time << " ms, cross node: " <<  gs -> opt.steps[i].cross_node << " ms, intra all2all: " << gs -> opt.steps[i].intra_all2all << std::endl;
        }
        std::cout << "=========================================================================" <<std::endl;   
    }
    
#endif
}

void run_scheduler(struct GlobalScheduler * gs){
    FastAll2All all2all;
    init_fastall2all(&all2all, &gs->mat);
    to_scaled_doubly_stochastic_matrix_fastall2all(&all2all);
    decompose_fastall2all(&all2all);
    // LOG("verify deccomposition: %u\n", verify_decomposition_fastall2all(&all2all));
    // if (gs -> sched->rankid ==0){
    //     std::cout << "[PSETS]: ";
    //     for (uint i = 0; i < all2all.p_sets_n; i++){
    //         std::cout << get_freq_permutation_set(&all2all.p_sets[all2all.p_sets_ascending[i]]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    uint pid = 0, lid = 0;

    /* Start Pipelining*/

    // generate schedule for intra-server all2all - balance first
    // balance once

    for (lid = 0; lid < gs->server_n; lid++){
        for (uint s = 0; s < gs->server_n; s++){
           if (s == gs->locals[lid]->server_id){
            continue;
           }
           uint src_svr = gs->locals[lid]->server_id;
           balance_one_server(gs->locals[lid], s, &gs->sched->balance[src_svr][s]);
        }
    }

    get_buffer_size(gs);

    // get intrinsic all-to-all
    for (lid = 0; lid < gs->server_n; lid++){
        uint src_svr = gs->locals[lid]->server_id;
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint64_t));
        // hipMemcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(TransferMatrixElement), hipMemcpyHostToHost);
    }

    uint step_id = 0;
    for (int i = all2all.p_sets_n - 1; i >= 0 ; i--){
        pid = all2all.p_sets_ascending[i];

        for (lid = 0; lid < gs->server_n; lid++){
            uint src_svr = gs->locals[lid]->server_id;
            uint dst_svr = 0;
            map_lookup(all2all.p_sets[pid].mp, all2all.p_sets[pid].mp_n, src_svr, &dst_svr);
            restore_one_server(gs->locals[lid],
                            dst_svr, &gs->sched->steps[step_id].channel[src_svr],
                            &gs->sched->steps[step_id].crossnode_sz[src_svr],
                            &gs->sched->steps[step_id + 1].restore[src_svr],
                            &gs->sched->steps[step_id + 1].restore_alltoall_sz[src_svr],
                            &gs->sched->steps[step_id + 1].direct_cpy[src_svr],
                            get_freq_permutation_set(&all2all.p_sets[pid]));
        }
        to_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].to_server);
        from_server_permutation_set(&all2all.p_sets[pid], gs->server_n, gs->sched->steps[step_id].from_server);
        step_id ++;
    }
    gs->sched->step_n = all2all.p_sets_n + 1;

    schedule_this_gpu(gs);

    uint data_size = 1;
    uint64_t workload_sz = 0;
    uint64_t max_cross_node_send_sz = 0, max_cross_node_recv_sz = 0;
    for (uint i = 0; i < gs->server_n; i++){
        uint64_t cross_node_send_sz = 0, cross_node_recv_sz = 0;
        for (uint j = 0; j < gs->server_n; j++){
            if (i == j) continue;
            cross_node_send_sz += gs->locals[i]->server2server_data[j];
            cross_node_recv_sz += gs->locals[j]->server2server_data[i];
        }
        max_cross_node_send_sz = MAX(cross_node_send_sz, max_cross_node_send_sz);
        max_cross_node_recv_sz = MAX(cross_node_recv_sz, max_cross_node_recv_sz);

        for (uint k = 0; k < gs->gpu_n; k++){
            for (uint z = 0; z <  gs->gpu_n * gs->server_n; z++){
                workload_sz += gs->locals[i]->data[k][z];
            }
        }
    }
    gs -> opt.workload_sz = workload_sz;
    gs -> opt.crossnode_time_limit = MAX(max_cross_node_send_sz, max_cross_node_recv_sz)  * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT; // ms
    gs -> opt.algbw_limit = workload_sz * data_size * 1e-6 / (gs->gpu_n * gs->server_n) / gs -> opt.crossnode_time_limit; //GBps

#if FLASH_PROFILE
    if (gs->gpu_sched->rankid == 0){
        gs -> opt.step_n = gs->sched->step_n;
        uint64_t * traffic_matrix = new uint64_t[gs -> gpu_n * gs -> gpu_n];
        gs -> opt.flash_total_time = 0;
        // load balance    
        gs -> opt.load_balance = 0;
        for (uint server_id = 0; server_id < gs -> server_n; server_id ++){
            memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
            for (uint i = 0; i < gs -> gpu_n; i ++){
                for (uint j = 0; j < gs -> gpu_n; j++){
                    if (i == j) continue;

                    for (uint dst_gpu = 0; dst_gpu < gs -> gpu_n; dst_gpu ++){
                        for (uint s = 0; s <  gs -> server_n; s++){
                            if (server_id == s)
                                continue;
                            traffic_matrix[i * gs -> gpu_n + j] += (gs -> sched -> balance)[server_id][s][i * gs -> gpu_n + j].sz[dst_gpu] * data_size;
                        }
                    }
                }
            }
            gs -> opt.load_balance_per_server[server_id] = alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6); //ms
            gs -> opt.load_balance = MAX(gs -> opt.load_balance, gs -> opt.load_balance_per_server[server_id]);
        }
        gs -> opt.flash_total_time += gs -> opt.load_balance;
        // first step and intrinsic alltoall
        // cross node
        uint64_t cross_node_max = 0;
        for (uint server_id = 0; server_id < gs->server_n; server_id ++){
            for (uint gpu_id = 0; gpu_id < gs->gpu_n; gpu_id++){
                cross_node_max = MAX(cross_node_max, (gs -> sched -> steps)[0].crossnode_sz[server_id][gpu_id]);
            }
        }
        gs -> opt.steps[0].cross_node = cross_node_max * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT;
        // intrinsic alltoall
        double intra_all2all_max_time = 0;
        for (uint server_id = 0; server_id < gs->server_n; server_id ++){
            memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
            for (uint i = 0; i < gs -> gpu_n; i ++){
                for (uint j = 0; j < gs -> gpu_n; j++){
                    if (i == j) continue;
                    traffic_matrix[i * gs -> gpu_n + j] = gs->locals[server_id]->intrinsic_all2all[i * gs -> gpu_n + j] * data_size;
                }
            }
            intra_all2all_max_time = MAX(intra_all2all_max_time, alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6));
        }
        gs -> opt.steps[0].intra_all2all = intra_all2all_max_time;
        gs -> opt.steps[0].step_time = MAX(gs -> opt.steps[0].intra_all2all, gs -> opt.steps[0].cross_node);
        gs -> opt.flash_total_time += gs -> opt.steps[0].step_time;

        // middle steps
        for (uint step_id = 1; step_id <= gs->sched->step_n - 1; step_id ++){
            gs -> opt.steps[step_id].cross_node = 0;
            if (step_id != gs->sched->step_n - 1){
                // cross_node
                cross_node_max = 0;
                for (uint server_id = 0; server_id < gs->server_n; server_id ++){
                    for (uint gpu_id = 0; gpu_id < gs->gpu_n; gpu_id++){
                        cross_node_max = MAX(cross_node_max, (gs -> sched -> steps)[step_id].crossnode_sz[server_id][gpu_id]);
                    }
                }
                gs -> opt.steps[step_id].cross_node = cross_node_max * data_size * 1e-6 / (double) INTER_SERVER_LINK_TPUT;
            }
            // intra alltoall for data redistribution
            intra_all2all_max_time = 0;
            for (uint server_id = 0; server_id < gs->server_n; server_id ++){
                memset(traffic_matrix, 0, gs -> gpu_n * gs -> gpu_n * sizeof(uint64_t));
                for (uint i = 0; i < gs -> gpu_n; i ++){
                    for (uint j = 0; j < gs -> gpu_n; j++){
                        if (i == j) continue;
                        traffic_matrix[i * gs -> gpu_n + j] =  (gs -> sched -> steps)[step_id].restore_alltoall_sz[server_id][i][j] * data_size;
                    }
                }
                intra_all2all_max_time = MAX(intra_all2all_max_time, alltoall_completion_time_with_spreadout(traffic_matrix, gs -> gpu_n, INTRA_SERVER_LINK_TPUT * 1e6));
            }
            gs -> opt.steps[step_id].intra_all2all = intra_all2all_max_time;
            gs -> opt.steps[step_id].step_time = MAX(gs -> opt.steps[step_id].intra_all2all, gs -> opt.steps[step_id].cross_node);
            gs -> opt.flash_total_time += gs -> opt.steps[step_id].step_time;
        }
        delete[] traffic_matrix;
        // -------------------------------------------------------------------------------
        // PRINT OUT PROFILING RESULT
        //--------------------------------------------------------------------------------
        std::cout << "==================FLASH THEORETICAL RESULT==============================" <<std::endl;
        std::cout << "Achievable throughput: " << gs -> opt.workload_sz * data_size * 1e-6 / (gs->gpu_n * gs->server_n) / gs -> opt.flash_total_time << " GBps" << std::endl
            << "Total completion time : " <<  gs -> opt.flash_total_time << " ms" << std::endl
            << "Load Balance: " << gs -> opt.load_balance << " ms" << std::endl
            << "Step n: " << gs->opt.step_n << std::endl
            << "Optimal cross node time: " << gs->opt.crossnode_time_limit <<  " ms" <<std::endl
            << "Throughput limit: " << gs->opt.algbw_limit << " GBps" << std::endl;

        std::cout << "----------------------------------------------------------------" <<std::endl;

        for (uint i = 0; i < gs->opt.step_n; i++){
            std::cout << "[Step " << i << "] Time: " <<  gs -> opt.steps[i].step_time << " ms, cross node: " <<  gs -> opt.steps[i].cross_node << " ms, intra all2all: " << gs -> opt.steps[i].intra_all2all << std::endl;
        }
        std::cout << "=========================================================================" <<std::endl;   
    }
    
#endif
}

double alltoall_completion_time_with_spreadout(uint64_t * workload, uint dim, double tput){
    double total_time = 0;
    for (uint i = 0; i < dim; i ++){
        uint64_t largest_diagnal = 0;
        for (uint j = 0; j < dim; j ++){
            largest_diagnal = MAX(largest_diagnal, workload[j * dim + ((j + i) % dim)]);
        }
        total_time += (double)largest_diagnal / tput;
    }
    return total_time;
}



#if ABLATION_TEST
void run_scheduler_ablation(struct GlobalScheduler * gs){
    FastAll2All all2all;
    init_fastall2all(&all2all, &gs->mat);
    to_scaled_doubly_stochastic_matrix_fastall2all(&all2all);
    decompose_fastall2all(&all2all);
    uint pid = 0, lid = 0;

    /* Start Pipelining*/

    // generate schedule for intra-server all2all - balance first
    // balance once

    for (lid = 0; lid < gs->server_n; lid++){
        for (uint s = 0; s < gs->server_n; s++){
           if (s == gs->locals[lid]->server_id){
            continue;
           }
           uint src_svr = gs->locals[lid]->server_id;
           balance_one_server(gs->locals[lid], s, &gs->sched->balance[src_svr][s]);
        }
    }

    get_buffer_size(gs);

    // get intrinsic all-to-all
    for (lid = 0; lid < gs->server_n; lid++){
        uint src_svr = gs->locals[lid]->server_id;
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint64_t));
        // hipMemcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(TransferMatrixElement), hipMemcpyHostToHost);
    }

    schedule_this_gpu_ablation(gs);
}


void schedule_this_gpu_ablation(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;

    // ------------------------------------------
    // Intrinsic alltoall: sendbuff => recvbuff
    //-------------------------------------------

    for (uint r = 0; r < gpu_n; r++){
        uint cur_gpu = server_id * gpu_n + r;
        uint64_t send_data_sz = gs -> buff_parameter -> sendbuff_sz[cur_gpu];
        if (send_data_sz > 0){
            uint64_t send_data_disp = gs -> buff_parameter -> sendbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].disp = send_data_disp;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].sz = send_data_sz;
            gs -> gpu_sched -> intrinsic_send_n ++;
        }

        uint64_t recv_data_sz = gs -> buff_parameter -> recvbuff_sz[cur_gpu];
        if (recv_data_sz > 0){
            uint64_t recv_data_disp = gs -> buff_parameter -> recvbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> intrinsic_recv_n ++;
        }
    }

    // --------------------------------------------------
    // Load balance:
    // First step: lbsend_buff ==> lbrecv_buff
    // Second step: lbrecv_buff ---(memcpy)--> sendbuff
    // --------------------------------------------------

    // first step
    for (uint r = 0; r < gpu_n; r++){
        uint64_t send_data_sz = gs -> buff_parameter -> lbsend_sz[r];
        uint cur_gpu_global_id = server_id * gpu_n + r;
        if (send_data_sz > 0){
            uint64_t send_data_disp = gs -> buff_parameter -> lbsend_disp[r];
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].disp = send_data_disp;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].sz = send_data_sz;
            gs -> gpu_sched -> balance_send_n ++;
        }
        uint64_t recv_data_sz = gs -> buff_parameter -> lbrecv_sz[r];
        if (recv_data_sz > 0){
            uint64_t recv_data_disp = gs -> buff_parameter -> lbrecv_disp[r];
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> balance_recv_n ++;
        }
    }
    // second step
    for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s < server_n; s ++){
                if (s == server_id){
                    continue;
                }
                uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                uint64_t cpy_sz = gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                if (cpy_sz > 0){
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].src_disp =
                        gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].dst_disp =
                        gs -> buff_parameter -> sendbuff_region[dst_gpu_global_id].src_gpu_disp[src_gpu];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].sz = cpy_sz;
                    gs -> gpu_sched -> balance_memcpy_n ++;
                }
            }
        }
    }


    // ---------------------------------------------------------
    // Cross node: sendbuff ==> crosbuff
    // ---------------------------------------------------------
    // only one crossnode send per dst server
    // crossbuff send
    for (uint dst_svr = 0; dst_svr < server_n; dst_svr ++){
        if (dst_svr == server_id) continue;
        uint64_t dst_svr_sendbuff_sz = 0;
        for (uint i = 0; i < gpu_n; i++){
            dst_svr_sendbuff_sz += gs -> buff_parameter -> sendbuff_sz[dst_svr * gpu_n + i];
        }
        if (dst_svr_sendbuff_sz > 0){
            gs -> gpu_sched -> ablation_crossnode_send[gs -> gpu_sched -> ablation_crossnode_send_n].sz = dst_svr_sendbuff_sz;
            gs -> gpu_sched -> ablation_crossnode_send[gs -> gpu_sched -> ablation_crossnode_send_n].gpu = dst_svr * gpu_n + local_rank_id;
            gs -> gpu_sched -> ablation_crossnode_send[gs -> gpu_sched -> ablation_crossnode_send_n].disp = gs -> buff_parameter -> sendbuff_disp[dst_svr * gpu_n];
            gs -> gpu_sched -> ablation_crossnode_send_n ++;
        }

    }

    // crossbuff recv
    gs -> buff_parameter -> ablation_crosbuff_total_sz = 0;
    for (uint src_svr = 0; src_svr < server_n; src_svr ++){
        if (src_svr == server_id) continue;
        uint64_t src_svr_sendbuff_sz = 0;
        for (uint i = 0; i < gpu_n; i++){
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                src_svr_sendbuff_sz += gs->locals[src_svr]->data_after_balance[local_rank_id][server_id * gpu_n + i].sz[src_gpu];
            }
        }
        if (src_svr_sendbuff_sz > 0){
            // crossbuff
            gs -> gpu_sched -> ablation_crossnode_recv[gs -> gpu_sched -> ablation_crossnode_recv_n].sz = src_svr_sendbuff_sz;
            gs -> gpu_sched -> ablation_crossnode_recv[gs -> gpu_sched -> ablation_crossnode_recv_n].gpu = src_svr * gpu_n + local_rank_id;
            gs -> gpu_sched -> ablation_crossnode_recv[gs -> gpu_sched -> ablation_crossnode_recv_n].disp = gs -> buff_parameter -> ablation_crosbuff_total_sz;
            gs -> gpu_sched -> ablation_crossnode_recv_n ++;
            gs -> buff_parameter -> ablation_crosbuff_total_sz += src_svr_sendbuff_sz;
        }
    }

    // restore data from crossnode transfer
    uint64_t ablation_restore_recvdisp[MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER], ablation_dcpy_disp[MAX_SERVER_NUM];
    memset(ablation_restore_recvdisp, 0, sizeof(uint64_t) * MAX_SERVER_NUM_TIMES_GPU_NUM_PER_SERVER);
    memset(ablation_dcpy_disp, 0, sizeof(uint64_t) * MAX_SERVER_NUM);
    uint64_t ablation_crossbuff_disp = 0, ablation_rstrbuff_disp = 0;
    for (uint src_svr = 0; src_svr < server_n; src_svr ++){
        if (src_svr == server_id) continue;
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            // restore send
            uint64_t send_gpu_sz = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                send_gpu_sz += gs -> locals[src_svr] -> data_after_balance[local_rank_id][server_id * gpu_n + local_gpu].sz[src_gpu];
            }
            if (local_gpu == local_rank_id){
                ablation_dcpy_disp[src_svr] = ablation_crossbuff_disp;
                ablation_crossbuff_disp += send_gpu_sz;
                continue;
            }

            if (send_gpu_sz > 0){
                gs -> gpu_sched -> ablation_restore_send[gs -> gpu_sched -> ablation_restore_send_n].sz = send_gpu_sz;
                gs -> gpu_sched -> ablation_restore_send[gs -> gpu_sched -> ablation_restore_send_n].gpu = server_id * gpu_n + local_gpu;
                gs -> gpu_sched -> ablation_restore_send[gs -> gpu_sched -> ablation_restore_send_n].disp = ablation_crossbuff_disp;
                gs -> gpu_sched -> ablation_restore_send_n ++;
                ablation_crossbuff_disp += send_gpu_sz;
            }
            // restore recv
            uint64_t recv_gpu_sz = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                recv_gpu_sz += gs -> locals[src_svr] -> data_after_balance[local_gpu][server_id * gpu_n + local_rank_id].sz[src_gpu];
            }
            if (recv_gpu_sz > 0){
                gs -> gpu_sched -> ablation_restore_recv[gs -> gpu_sched -> ablation_restore_recv_n].sz = recv_gpu_sz;
                gs -> gpu_sched -> ablation_restore_recv[gs -> gpu_sched -> ablation_restore_recv_n].gpu = server_id * gpu_n + local_gpu;
                gs -> gpu_sched -> ablation_restore_recv[gs -> gpu_sched -> ablation_restore_recv_n].disp = ablation_rstrbuff_disp;
                ablation_restore_recvdisp[src_svr * gpu_n + local_gpu] = ablation_rstrbuff_disp;
                gs -> gpu_sched -> ablation_restore_recv_n ++;
                ablation_rstrbuff_disp += recv_gpu_sz;
            }
        }
    }
    gs -> buff_parameter -> ablation_rstrbuff_total_sz = ablation_rstrbuff_disp;

    // direct memcpy
    for (uint src_svr = 0; src_svr < server_n; src_svr ++){
        if (src_svr == server_id) continue;
        uint64_t ablation_direct_cpy_src_disp = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint64_t cpy_sz = gs -> locals[src_svr] -> data_after_balance[local_rank_id][server_id * gpu_n + local_rank_id].sz[src_gpu];
            if (cpy_sz > 0){
                uint cur_src_gpu_global_id = src_svr * gpu_n + src_gpu;
                gs -> gpu_sched ->ablation_direct_memcpy[gs -> gpu_sched -> ablation_direct_memcpy_n].sz = cpy_sz;
                gs -> gpu_sched ->ablation_direct_memcpy[gs -> gpu_sched -> ablation_direct_memcpy_n].src_disp = ablation_dcpy_disp[src_svr] + ablation_direct_cpy_src_disp;
                gs -> gpu_sched ->ablation_direct_memcpy[gs -> gpu_sched -> ablation_direct_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + gs -> locals[src_svr] -> data_after_balance[local_rank_id][server_id * gpu_n + local_rank_id].offset[src_gpu];
                ablation_direct_cpy_src_disp += cpy_sz;
                gs -> gpu_sched -> ablation_direct_memcpy_n ++;
            }
        }
    }

    // restore memcpy
    for (uint src_svr = 0; src_svr < server_n; src_svr++){
        if (src_svr == server_id) continue;
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            if (local_gpu == local_rank_id) continue;
            uint64_t ablation_restore_cpy_src_disp = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                uint64_t cpy_sz = gs -> locals[src_svr] -> data_after_balance[local_gpu][server_id * gpu_n + local_rank_id].sz[src_gpu];
                if(cpy_sz > 0){
                    uint cur_src_gpu_global_id = src_svr * gpu_n + src_gpu;
                    gs -> gpu_sched -> ablation_restore_memcpy[gs -> gpu_sched -> ablation_restore_memcpy_n].sz = cpy_sz;
                    gs -> gpu_sched -> ablation_restore_memcpy[gs -> gpu_sched -> ablation_restore_memcpy_n].src_disp = ablation_restore_recvdisp[src_svr * gpu_n + local_gpu] + ablation_restore_cpy_src_disp;
                    gs -> gpu_sched -> ablation_restore_memcpy[gs -> gpu_sched -> ablation_restore_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] +  gs -> locals[src_svr] -> data_after_balance[local_gpu][server_id * gpu_n + local_rank_id].offset[src_gpu];
                    ablation_restore_cpy_src_disp += cpy_sz;
                    gs -> gpu_sched -> ablation_restore_memcpy_n ++;
                }
            }
        }
    }
}

#endif

void get_flash_buffer_size(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;
    
    gs->flash_buffer_sz_params->balance_send_param.total_sz = 0;
    gs->flash_buffer_sz_params->balance_recv_param.total_sz = 0;
    //--------------------------------------------------
    // LB send buff: |        local gpu 0               | local gpu 1 | ... |
    //               |dst gpu 0 | dst gpu 1 | dst gpu 2 | ...
    //               | s0  | s1 | s0  | s1  | ... |
    // --------------------------------------------------
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
        gs->flash_buffer_sz_params->balance_send_param.disp[local_gpu] = gs->flash_buffer_sz_params->balance_send_param.total_sz;
        gs->flash_buffer_sz_params->balance_recv_param.disp[local_gpu] = gs->flash_buffer_sz_params->balance_recv_param.total_sz;
        uint64_t total_send_sz = 0, total_recv_sz = 0;
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s != server_n; s++){
                if (s == server_id){
                    continue;
                }
                uint global_rank = s * gpu_n + dst_gpu;
                uint64_t send_data_sz = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[dst_gpu];
                if (send_data_sz > 0){
                    gs->flash_buffer_sz_params->balance_send_param.layout[local_gpu].disp[global_rank] = gs->flash_buffer_sz_params->balance_send_param.total_sz;
                    gs->flash_buffer_sz_params->balance_send_param.layout[local_gpu].sz[global_rank] = send_data_sz;
                    gs->flash_buffer_sz_params->balance_send_param.layout[local_gpu].data_offset[global_rank] = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].offset[dst_gpu];
                    gs->flash_buffer_sz_params->balance_send_param.layout[local_gpu].rank_n ++;
                    gs->flash_buffer_sz_params->balance_send_param.total_sz += send_data_sz;
                }
                total_send_sz += send_data_sz;

                uint64_t recv_data_sz = (gs -> sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[dst_gpu];
                if (recv_data_sz > 0){
                    gs->flash_buffer_sz_params->balance_recv_param.layout[local_gpu].disp[global_rank] = gs->flash_buffer_sz_params->balance_recv_param.total_sz;
                    gs->flash_buffer_sz_params->balance_recv_param.layout[local_gpu].sz[global_rank] = recv_data_sz;
                    gs->flash_buffer_sz_params->balance_recv_param.layout[local_gpu].rank_n ++;
                    gs->flash_buffer_sz_params->balance_recv_param.total_sz += recv_data_sz;
                }
                total_recv_sz += recv_data_sz;
            }
        }
        gs->flash_buffer_sz_params->balance_send_param.sz[local_gpu] = total_send_sz;
        gs->flash_buffer_sz_params->balance_recv_param.sz[local_gpu] = total_recv_sz;
    }
    //create additional spaces at recv buffer to signal completion of load balance
    gs->flash_buffer_sz_params->balance_recv_param.total_sz += sizeof(uint64_t) * gpu_n;

    // Send buffer, not including data that are destinated to the same node
    //  |                dst rank 0                     | dst rank 1 | ... | dst rank n |
    //  | local gpu 0 | local gpu 1 | ... | local gpu n | ...
    gs->flash_buffer_sz_params->send_param.total_sz = 0;
    for (uint i = 0; i < server_n * gpu_n; i ++){
        if (i / gpu_n == server_id){
            continue;
        }
        gs->flash_buffer_sz_params->send_param.disp[i] = gs->flash_buffer_sz_params->send_param.total_sz;
        uint64_t send_buff_sz = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint64_t lb_data_sz = gs->locals[server_id]->data_after_balance[local_rank_id][i].sz[src_gpu];
            if (lb_data_sz > 0){
                gs->flash_buffer_sz_params->send_param.layout[i].src_gpu_disp[src_gpu] = gs->flash_buffer_sz_params->send_param.total_sz;
                gs->flash_buffer_sz_params->send_param.layout[i].src_gpu_sz[src_gpu] = lb_data_sz;
                gs->flash_buffer_sz_params->send_param.layout[i].data_offset[src_gpu] = gs->locals[server_id]->data_after_balance[local_rank_id][i].offset[src_gpu];
                gs->flash_buffer_sz_params->send_param.total_sz += lb_data_sz;
                send_buff_sz += lb_data_sz;
                gs->flash_buffer_sz_params->send_param.layout[i].src_gpu_n ++;
            }
        }
        gs->flash_buffer_sz_params->send_param.sz[i] = send_buff_sz;
    }
    gs->flash_buffer_sz_params->send_param.max_total_sz = 0;
    for (uint s = 0; s < server_n; s++){
        for (uint local_rank = 0; local_rank < gpu_n; local_rank ++){
            uint64_t current_send_buffer_sz = 0;
            for (uint i = 0; i < server_n * gpu_n; i ++){
                if (i / gpu_n == s){
                    continue;
                }
                for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                    current_send_buffer_sz += gs->locals[s]->data_after_balance[local_rank][i].sz[src_gpu];
                }
            }
            gs->flash_buffer_sz_params->send_param.max_total_sz = MAX(gs->flash_buffer_sz_params->send_param.max_total_sz, current_send_buffer_sz);
        }
    }

    // Output buffer
    // |     src rank 0    | src rank 1 | ... | src rank n |
    gs->flash_buffer_sz_params->out_total_sz = 0;
    for (uint i = 0; i < server_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            uint src_rank = i * gpu_n + j;
            gs->flash_buffer_sz_params->out_disp_per_rank[src_rank] = gs->flash_buffer_sz_params->out_total_sz;
            gs->flash_buffer_sz_params->out_sz_per_rank[src_rank] = gs->locals[i]->data[j][global_rank_id];
            gs->flash_buffer_sz_params->out_total_sz += gs->locals[i]->data[j][global_rank_id];
        }
    }

    // Input buffer
    // |        dst rank 0       | dst rank 1 | ... | dst rank
    gs->flash_buffer_sz_params->in_total_sz = 0;
    for (uint i = 0; i < server_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            uint dst_rank = i * gpu_n + j;
            gs->flash_buffer_sz_params->in_disp_per_rank[dst_rank] = gs->flash_buffer_sz_params->in_total_sz;
            gs->flash_buffer_sz_params->in_sz_per_rank[dst_rank] = gs->locals[server_id]->data[local_rank_id][dst_rank];
            gs->flash_buffer_sz_params->in_total_sz += gs->locals[server_id]->data[local_rank_id][dst_rank];
        }
    }
}

void get_buffer_size(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;


    gs->buff_parameter->lbsend_total_sz = 0;
    gs->buff_parameter->lbrecv_total_sz = 0;
    //--------------------------------------------------
    // LB send buff: |        local gpu 0               | local gpu 1 | ... |
    //               |dst gpu 0 | dst gpu 1 | dst gpu 2 | ...
    //               | s0  | s1 | s0  | s1  | ... |
    // --------------------------------------------------
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){

        gs->buff_parameter->lbsend_disp[local_gpu] = gs->buff_parameter->lbsend_total_sz;
        gs->buff_parameter->lbrecv_disp[local_gpu] = gs->buff_parameter->lbrecv_total_sz;
        uint64_t send_area_sz = 0, recv_area_sz = 0;
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){

            gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_disp[dst_gpu] = gs->buff_parameter->lbsend_total_sz;
            uint64_t send_region_sz = 0, recv_region_sz = 0;
            bool send_lb = false, recv_lb = false;
            for (uint s = 0; s != server_n; s++){
                if (s == server_id){
                    continue;
                }
                uint64_t send_data_sz = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[dst_gpu];
                if (send_data_sz > 0){
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = gs->buff_parameter->lbsend_total_sz;
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = send_data_sz;
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s] = (gs -> sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].offset[dst_gpu];;
                    gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    gs->buff_parameter->lbsend_total_sz += send_data_sz;
                    send_region_sz += send_data_sz;
                    send_lb = true;
                }

                uint64_t recv_data_sz = (gs -> sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[dst_gpu];
                if (recv_data_sz > 0){
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s] = gs->buff_parameter->lbrecv_total_sz;
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s] = recv_data_sz;
                    gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_region[dst_gpu].server_n ++;
                    gs->buff_parameter->lbrecv_total_sz += recv_data_sz;
                    recv_region_sz += recv_data_sz;
                    recv_lb = true;
                }
            }
            gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_sz[dst_gpu] = send_region_sz;
            if (send_lb) gs->buff_parameter->lbsend_area[local_gpu].dst_gpu_n ++;
            send_area_sz += send_region_sz;
            gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_sz[dst_gpu] = recv_region_sz;
            if (recv_lb) gs->buff_parameter->lbrecv_area[local_gpu].dst_gpu_n ++;
            recv_area_sz += recv_region_sz;

        }
        gs->buff_parameter->lbsend_sz[local_gpu] = send_area_sz;
        gs->buff_parameter->lbrecv_sz[local_gpu] = recv_area_sz;
    }
    // gs->buff_parameter->lbsend_total_sz = mem_align(gs->buff_parameter->lbsend_total_sz);
    // gs->buff_parameter->lbrecv_total_sz = mem_align(gs->buff_parameter->lbrecv_total_sz);


    gs->buff_parameter->sendbuff_total_sz = 0;
    for (uint i = 0; i < server_n * gpu_n; i ++){
        gs->buff_parameter->sendbuff_disp[i] = gs->buff_parameter->sendbuff_total_sz;
        uint64_t send_buff_sz = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint64_t lb_data_sz = gs->locals[server_id]->data_after_balance[local_rank_id][i].sz[src_gpu];
            if (lb_data_sz > 0){
                gs->buff_parameter->sendbuff_region[i].src_gpu_disp[src_gpu] = gs->buff_parameter->sendbuff_total_sz;
                gs->buff_parameter->sendbuff_region[i].src_gpu_sz[src_gpu] = lb_data_sz;
                gs->buff_parameter->sendbuff_region[i].src_gpu_offset[src_gpu] = gs->locals[server_id]->data_after_balance[local_rank_id][i].offset[src_gpu];;
                gs->buff_parameter->sendbuff_total_sz += lb_data_sz;
                send_buff_sz += lb_data_sz;
                gs->buff_parameter->sendbuff_region[i].src_gpu_n ++;
            }
        }
        gs->buff_parameter->sendbuff_sz[i] = send_buff_sz;
    }
    // gs->buff_parameter->sendbuff_total_sz = mem_align(gs->buff_parameter->sendbuff_total_sz);

    gs->buff_parameter->recvbuff_total_sz = 0;
    for (uint i = 0; i < server_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            uint src_rank = i * gpu_n + j;
            gs->buff_parameter->recvbuff_disp[src_rank] = gs->buff_parameter->recvbuff_total_sz;
            gs->buff_parameter->recvbuff_sz[src_rank] = gs->locals[i]->data[j][global_rank_id];
            gs->buff_parameter->recvbuff_total_sz += gs->locals[i]->data[j][global_rank_id];
        }
    }
    // gs->buff_parameter->recvbuff_total_sz = mem_align(gs->buff_parameter->recvbuff_total_sz);

    gs->buff_parameter->inputbuff_total_sz = 0;
    for (uint i = 0; i < server_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            uint dst_rank = i * gpu_n + j;
            gs->buff_parameter->inputbuff_disp[dst_rank] = gs->buff_parameter->inputbuff_total_sz;
            gs->buff_parameter->inputbuff_sz[dst_rank] = gs->locals[server_id]->data[local_rank_id][dst_rank];
            gs->buff_parameter->inputbuff_total_sz += gs->locals[server_id]->data[local_rank_id][dst_rank];
        }
    }

}

void assign_intra_alltoall_to_blocks(struct device_intra_alltoall_params_t * param, uint64_t * src_offset, uint64_t * dst_offset, uint64_t * sz, uint local_rank_n, uint block_n, uint64_t block_tx_sz, uint data_size){
    for (uint r = 0; r < local_rank_n; r++){
        uint64_t send_sz_to_rank = sz[r];
        if (send_sz_to_rank == 0) continue;
        uint64_t send_sz_per_block = CEIL_DIV(send_sz_to_rank, block_n);
        uint64_t already_sent_sz = 0;
        for (uint b = 0; b < block_n; b++){
            if (already_sent_sz >= send_sz_to_rank) break;
            uint64_t actual_send_sz_per_block = std::min(send_sz_per_block, send_sz_to_rank - already_sent_sz);
            param -> src_disp[r * block_n + b] = src_offset[r] + already_sent_sz; // offset in the sender buffer for each block
            param -> dst_disp[r * block_n + b] = dst_offset[r] + already_sent_sz; // offset in the receiver buffer for each block
            param -> sz[r * block_n + b] = actual_send_sz_per_block;              // block total transfer size    
            already_sent_sz += actual_send_sz_per_block;
            param -> round_n[r * block_n + b] = CEIL_DIV(param -> sz[r * block_n + b] * data_size, block_tx_sz);
            // if (param -> src_disp[r * block_n + b] % 16 != 0 || param -> dst_disp[r * block_n + b] % 16 != 0){
            //     printf("MISALIGN: INTRA-AlltoAll, block: %u, in disp: %lu, in disp hex: %lx, out disp: %lu, out disp hex: %lx, sz: %lu, sz hex: %lx\n", b, param -> src_disp[r * block_n + b], param -> src_disp[r * block_n + b], param -> dst_disp[r * block_n + b], param -> dst_disp[r * block_n + b], actual_send_sz_per_block, actual_send_sz_per_block);
            // }
        }
    }   
}


void assign_intra_redistribute_to_blocks(struct device_intra_redistribute_alltoall_params_t * param, uint64_t * src_offset, uint64_t * dst_offset, uint64_t * sz, uint local_rank_n, uint block_n, uint64_t block_tx_sz, uint data_size){
    for (uint dst_gpu = 0; dst_gpu < local_rank_n; dst_gpu ++){
        for (uint src_gpu = 0; src_gpu < local_rank_n; src_gpu ++){
            uint64_t send_sz_from_src_to_dst = sz[dst_gpu * local_rank_n + src_gpu];
            if (send_sz_from_src_to_dst == 0)
                continue;
            uint64_t send_sz_per_block = CEIL_DIV(send_sz_from_src_to_dst, block_n);
            uint64_t already_sent_sz = 0;
            for (uint b = 0; b < block_n; b++){
                if (already_sent_sz >= send_sz_from_src_to_dst) break;
                uint64_t actual_send_sz_per_block = MIN(send_sz_per_block, send_sz_from_src_to_dst - already_sent_sz);
                param -> src_disp[(dst_gpu * local_rank_n + src_gpu) * block_n + b] = src_offset[dst_gpu * local_rank_n + src_gpu] + already_sent_sz; // offset in the sender buffer for each block
                param -> dst_disp[(dst_gpu * local_rank_n + src_gpu) * block_n + b] = dst_offset[dst_gpu * local_rank_n + src_gpu] + already_sent_sz; // offset in the receiver buffer for each block
                param -> sz[(dst_gpu * local_rank_n + src_gpu) * block_n + b] = actual_send_sz_per_block;              // block total transfer size    
                already_sent_sz += actual_send_sz_per_block;
                param -> round_n[(dst_gpu * local_rank_n + src_gpu) * block_n + b] =  CEIL_DIV(actual_send_sz_per_block * data_size, block_tx_sz);
            }
        }
    }
}

void plan_chunk_pipeline(struct chunk_metadata_t * param, uint64_t p2p_sz){
    if (p2p_sz == 0){
        param -> chunk_n = 0;
        param -> chunk_ready_threshold[0] = 0;
        param -> chunk_sz[0] = 0;
        param -> chunk_disp[0] = 0;
    }else if (p2p_sz < FLASH_MIN_CHUNK_SIZE){
        // no need for chunk pipelining if the transfer size is too small, just one chunk is enough
        param -> chunk_n = 1;
        param -> chunk_ready_threshold[0] = p2p_sz;
        param -> chunk_sz[0] = p2p_sz;
        param -> chunk_disp[0] = 0;
    }else{
        uint64_t chunk = FLASH_CHUNK_ALIGN(CEIL_DIV(p2p_sz, FLASH_MAX_CHUNK_NUM));    // round up to a multiple of 4MB
        chunk = MAX(FLASH_MIN_CHUNK_SIZE, chunk);   // we don't want the chunk too small or too many trunks
        uint64_t sent = 0;
        param -> chunk_n = 0;
        while (sent < p2p_sz) {
            uint64_t n = MIN(chunk, p2p_sz - sent);
            param -> chunk_disp[param->chunk_n] = sent;
            param -> chunk_sz[param->chunk_n] = n;
            param -> chunk_ready_threshold[param->chunk_n] = n + sent;
            param -> chunk_n ++;
            sent += n;
        }
    }
}


void flash_schedule_this_gpu(struct GlobalScheduler * gs){
    uint global_rank_id = gs->flash_sched->info.this_rank,
        local_rank_id = gs->flash_sched->info.this_rank % gs->flash_sched->info.local_rank_n,
        server_id = gs->flash_sched->info.server_id,
        server_n = gs->flash_sched->info.server_n,
        gpu_n = gs->flash_sched->info.local_rank_n,
        rank_n = gs->flash_sched->info.rank_n;

    uint64_t temp_intra_sz[GPU_NUM_PER_SERVER], temp_intra_src_disp[GPU_NUM_PER_SERVER], temp_intra_dst_disp[GPU_NUM_PER_SERVER], 
    temp_intra_redistribute_sz[GPU_NUM_PER_SERVER_SQUARE], temp_intra_redistribute_src_disp[GPU_NUM_PER_SERVER_SQUARE], temp_intra_redistribute_dst_disp[GPU_NUM_PER_SERVER_SQUARE];
    memset(&temp_intra_sz, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);
    memset(&temp_intra_src_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);
    memset(&temp_intra_dst_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);

    // -------------------------------------------------------------------------
    // Intrinsic alltoall: buffer_in => buffer_out, using Memory IPC Handle
    //--------------------------------------------------------------------------
    for (uint r = 0; r < gpu_n; r++){
        uint cur_gpu = server_id * gpu_n + r;
        temp_intra_sz[r] = gs -> flash_buffer_sz_params -> in_sz_per_rank[cur_gpu];
        temp_intra_src_disp[r] = gs -> flash_buffer_sz_params -> in_disp_per_rank[cur_gpu];

        uint64_t dst_disp_in_dst_rank = 0;
        for (uint i = 0; i < global_rank_id; i++){
            dst_disp_in_dst_rank += gs->locals[i / gpu_n]->data[i % gpu_n][cur_gpu];
        }
        temp_intra_dst_disp[r] = dst_disp_in_dst_rank;
    }
    assign_intra_alltoall_to_blocks(gs -> flash_sched -> host_intrinsic_alltoall, temp_intra_src_disp, temp_intra_dst_disp, temp_intra_sz, gpu_n,  gs->flash_sched->info.block_n, gs->flash_sched->info.min_tx_sz_at_a_time, gs->data_size);
    // copy intrinsic alltoall  metadata to device
    CUDACHECK(cudaMemcpy(gs -> flash_sched -> dev_intrinsic_alltoall, gs -> flash_sched -> host_intrinsic_alltoall, sizeof(struct device_intra_alltoall_params_t), cudaMemcpyHostToDevice));
    
    // if (global_rank_id == 0){
    //     for (uint r = 0; r < gpu_n; r++){
    //         printf("rank %u\n", r);
    //         for (uint b = 0; b < gs->flash_sched->info.block_n; b++){
    //             printf("\tblock-%u, src disp: %lu, dst disp: %lu, sz: %lu, rounds: %lu\n", b, gs -> flash_sched -> host_intrinsic_alltoall -> src_disp[r * gs->flash_sched->info.block_n +b], gs -> flash_sched -> host_intrinsic_alltoall -> dst_disp[r * gs->flash_sched->info.block_n +b], gs -> flash_sched -> host_intrinsic_alltoall -> sz[r * gs->flash_sched->info.block_n + b], gs -> flash_sched -> host_intrinsic_alltoall -> round_n[r * gs->flash_sched->info.block_n + b]);
    //         }
    //     }
    // }
    
    // -----------------------------------------------------------
    // Load balance:
    // First step: lbsend_buff ==> lbrecv_buff
    // Second step: lbrecv_buff ---(memcpy)--> sendbuff (nvshmem)
    // -----------------------------------------------------------
    // first step
    memset(&temp_intra_sz, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);
    memset(&temp_intra_src_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);
    memset(&temp_intra_dst_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER);
    for (uint r = 0; r < gpu_n; r++){
        temp_intra_sz[r] = gs -> flash_buffer_sz_params -> balance_send_param.sz[r];
        temp_intra_src_disp[r] = gs -> flash_buffer_sz_params -> balance_send_param.disp[r];

        uint64_t dst_disp_in_dst_rank = 0;
        for (uint local_gpu = 0; local_gpu < local_rank_id; local_gpu++){
            for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
                for (uint s = 0; s < server_n; s++){
                    if (s == server_id){
                        continue;
                    }
                    dst_disp_in_dst_rank += (gs -> sched -> balance)[server_id][s][local_gpu * gpu_n + r].sz[dst_gpu];
                }
            }
        }
        temp_intra_dst_disp[r] = dst_disp_in_dst_rank;
        if (gs -> flash_buffer_sz_params -> balance_recv_param.sz[r] > 0){
            gs -> flash_sched -> host_balance_alltoall -> recv_sz[r] = gs -> flash_buffer_sz_params -> balance_recv_param.sz[r];
        }
    }
    // printf("\n==================LB SEND===========================\n");
    // for (uint r = 0; r < gpu_n; r++){
    //     if (temp_intra_sz[r] > 0) printf("LBSEND: rank %u => rank %u, lbsend src disp: %lu, lbrecv dst disp: %lu, sz: %lu\n", local_rank_id, r, temp_intra_src_disp[r], temp_intra_dst_disp[r], temp_intra_sz[r]);
    // }
    
    assign_intra_alltoall_to_blocks(gs -> flash_sched -> host_balance_alltoall, temp_intra_src_disp, temp_intra_dst_disp, temp_intra_sz, gpu_n, gs->flash_sched->info.block_n, gs->flash_sched->info.min_tx_sz_at_a_time, gs->data_size);
    // copy balance alltoall  metadata to device
    CUDACHECK(cudaMemcpy(gs -> flash_sched -> dev_balance_alltoall, gs -> flash_sched -> host_balance_alltoall, sizeof(struct device_intra_alltoall_params_t), cudaMemcpyHostToDevice));

    // second step
    for (uint src_local_gpu = 0; src_local_gpu < gpu_n; src_local_gpu ++){
        for (uint dst_rank = 0; dst_rank < rank_n; dst_rank ++){
            if (dst_rank / gpu_n == server_id){
                continue;
            }
            uint64_t copy_sz = gs -> flash_buffer_sz_params -> balance_recv_param.layout[src_local_gpu].sz[dst_rank];
            if (copy_sz > 0){
                gs -> flash_sched -> balance_memcpy[gs -> flash_sched -> balance_memcpy_n].src_disp = gs -> flash_buffer_sz_params -> balance_recv_param.layout[src_local_gpu].disp[dst_rank];
                gs -> flash_sched -> balance_memcpy[gs -> flash_sched -> balance_memcpy_n].dst_disp = gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_disp[src_local_gpu];
                gs -> flash_sched -> balance_memcpy[gs -> flash_sched -> balance_memcpy_n].sz = copy_sz;
                gs -> flash_sched -> balance_memcpy_n ++;
            }
        }
    }

    // if (global_rank_id == 2){
    //     printf("\n======================SEND BUFFER========================\n");
    //     for (uint dst_rank = 0; dst_rank < rank_n; dst_rank ++){
    //         printf("dst rank %u\n", dst_rank);
    //         for (uint src_local_rank = 0; src_local_rank < gpu_n; src_local_rank ++){
    //             if (gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_sz[src_local_rank] > 0){
    //                 printf("\tsrc local gpu: %u, from %lu to %lu, sz: %lu\n", src_local_rank, gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_disp[src_local_rank], gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_disp[src_local_rank] + gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_sz[src_local_rank], gs -> flash_buffer_sz_params -> send_param.layout[dst_rank].src_gpu_sz[src_local_rank]);
    //             }
    //         }
    //     }
    //     printf("\n======================BALANCE COPY========================\n");
    //     for (uint copy_id = 0; copy_id < gs -> flash_sched -> balance_memcpy_n; copy_id++){
    //         printf("copy from balance recv buffer disp %lu to send buffer  disp %lu, sz: %lu\n", gs -> flash_sched -> balance_memcpy[copy_id].src_disp, gs -> flash_sched -> balance_memcpy[copy_id].dst_disp, gs -> flash_sched -> balance_memcpy[copy_id].sz);
    //     }
    // }

    // ---------------------------------------------------------
    // Internode node transer: sendbuff ==> internode_recvbuff
    // ---------------------------------------------------------
    // calculate buffer size of internode_recvbuff
    uint64_t max_pingpong_sz = 0;
    for (uint step_id = 0; step_id < gs->sched->step_n - 1; step_id++){
        for (uint src_svr = 0; src_svr < gs->server_n; src_svr++){
            max_pingpong_sz = MAX(max_pingpong_sz, (gs -> sched -> steps)[step_id].crossnode_sz[src_svr][local_rank_id]);
        }
    }
    max_pingpong_sz  = mem_align(max_pingpong_sz);
    gs -> flash_buffer_sz_params -> pingpong_sz = max_pingpong_sz;

    uint64_t sync1_threshold[rank_n], sync2_threshold[rank_n], * cur_sync_threshold;
    memset(sync1_threshold, 0, sizeof(uint64_t) * rank_n);
    memset(sync2_threshold, 0, sizeof(uint64_t) * rank_n);
    //-----------------------------------------------------------------
    // Pipeline stages
    //----------------------------------------------------------------
    gs -> flash_sched -> stage_n = gs -> sched -> step_n;
    uint step_id = 0;

    // first pipeline stage
    uint64_t crossnode_already_send_sz[MAX_SERVER_NUM];
    memset(crossnode_already_send_sz, 0 , sizeof(uint64_t) * MAX_SERVER_NUM);
    struct scheduling_step_t * cur_step = &(gs -> sched -> steps)[0];
    uint dst_server = cur_step -> to_server[server_id];
    uint src_server = cur_step -> from_server[server_id];
    uint dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    uint src_gpu_global_id = src_server * gpu_n + local_rank_id;

    gs -> flash_sched -> host_stages_internode[0].send_sz = cur_step -> crossnode_sz[server_id][local_rank_id];
    gs -> flash_sched -> host_stages_internode[0].dst_rank = dst_gpu_global_id;
    gs -> flash_sched -> host_stages_internode[0].src_disp = gs -> flash_buffer_sz_params -> send_param.disp[dst_server * gpu_n] + crossnode_already_send_sz[dst_server];
    crossnode_already_send_sz[dst_server] += gs -> flash_sched -> host_stages_internode[0].send_sz;

    gs -> flash_sched -> host_stages_internode[0].recv_sz = cur_step -> crossnode_sz[src_server][local_rank_id];
    gs -> flash_sched -> host_stages_internode[0].src_rank = src_gpu_global_id;
    gs -> flash_sched -> host_stages_internode[0].dst_disp = 0;

    sync1_threshold[gs -> flash_sched -> host_stages_internode[0].src_rank] += gs -> flash_sched -> host_stages_internode[0].recv_sz;
    gs -> flash_sched -> host_sync_threshold[0] = sync1_threshold[gs -> flash_sched -> host_stages_internode[0].src_rank];


    plan_chunk_pipeline(&gs -> flash_sched -> host_stages_send_chunks[0], gs -> flash_sched -> host_stages_internode[0].send_sz);

    // middle pipeline stages
    uint prev_dst_server = dst_server, prev_src_server = src_server;
    uint64_t redistribute_sz = 0;

    for (step_id = 1; step_id < gs->sched->step_n - 1; step_id++){
        cur_step = &(gs -> sched -> steps)[step_id];
        dst_server = cur_step -> to_server[server_id];
        src_server = cur_step -> from_server[server_id];
        dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
        src_gpu_global_id = src_server * gpu_n + local_rank_id;

        // internode transfer
        gs -> flash_sched -> host_stages_internode[step_id].send_sz = cur_step -> crossnode_sz[server_id][local_rank_id];
        gs -> flash_sched -> host_stages_internode[step_id].dst_rank = dst_gpu_global_id;
        gs -> flash_sched -> host_stages_internode[step_id].src_disp = gs -> flash_buffer_sz_params -> send_param.disp[dst_server * gpu_n] + crossnode_already_send_sz[dst_server];
        crossnode_already_send_sz[dst_server] += gs -> flash_sched -> host_stages_internode[step_id].send_sz;

        gs -> flash_sched -> host_stages_internode[step_id].recv_sz = cur_step -> crossnode_sz[src_server][local_rank_id];
        gs -> flash_sched -> host_stages_internode[step_id].src_rank = src_gpu_global_id;
        gs -> flash_sched -> host_stages_internode[step_id].dst_disp = 0;

        cur_sync_threshold = (step_id % 2 == 0) ? sync1_threshold : sync2_threshold;
        cur_sync_threshold[gs -> flash_sched -> host_stages_internode[step_id].src_rank] += gs -> flash_sched -> host_stages_internode[step_id].recv_sz;
        gs -> flash_sched -> host_sync_threshold[step_id] = cur_sync_threshold[gs -> flash_sched -> host_stages_internode[step_id].src_rank];
    
        plan_chunk_pipeline(&gs -> flash_sched -> host_stages_send_chunks[step_id], gs -> flash_sched -> host_stages_internode[step_id].send_sz);
        // internode recvbuffer, same as send buffer
        //  |                 dst rank 0                        | dst rank 1 | ... | dst rank n |
        //  | src local gpu 0 | local gpu 1 | ... | local gpu n | ...

        // Data redistribution for previous completed pipeline stage: internode_recvbuffer ==> buffer out via intranode alltoall
        memset(&temp_intra_redistribute_src_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
        memset(&temp_intra_redistribute_sz, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
        memset(&temp_intra_redistribute_dst_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
        redistribute_sz = 0;
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu++){ // dst gpu id in the current server
            uint64_t reditribute_send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][dst_gpu];
            if (local_rank_id == dst_gpu){
                // get metadata from direct copy
                uint64_t local_data_disp = 0;;
                for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){    // actual data source in the previous server
                    uint64_t local_data_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
                    if (local_data_sz > 0){
                        temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu] = redistribute_sz + local_data_disp;
                        uint cur_src_gpu_global_id = prev_src_server * gpu_n + src_gpu;
                        temp_intra_redistribute_sz[local_rank_id * gpu_n + src_gpu] = local_data_sz;
                        temp_intra_redistribute_dst_disp[local_rank_id * gpu_n + src_gpu] = gs -> flash_buffer_sz_params -> out_disp_per_rank[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset;
                        local_data_disp += local_data_sz;
                        // if (global_rank_id == 0){
                        //     printf("[STEP %u] RANK %u - DIRECT CPY: src rank: %u, src disp: %lu, dst disp: %lu, sz: %lu\n", step_id, global_rank_id, cur_src_gpu_global_id, temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_dst_disp[local_rank_id * gpu_n + src_gpu], temp_intra_redistribute_sz[local_rank_id * gpu_n + src_gpu]);  
                        // }
                    }
                }
                redistribute_sz += reditribute_send_data_sz;
            }else{
                uint64_t local_data_disp = 0;
                for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){    // actual data source in the previous server
                    uint64_t local_data_sz = cur_step -> restore[prev_src_server][local_rank_id][dst_gpu * gpu_n + src_gpu].sz;
                    if (local_data_sz > 0){
                        temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu] = redistribute_sz + local_data_disp;
                        uint cur_src_gpu_global_id = prev_src_server * gpu_n + src_gpu;
                        temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu] = local_data_sz;
                        uint64_t dst_disp_in_dst_rank = 0;
                        for (uint i = 0; i < cur_src_gpu_global_id; i++){
                            dst_disp_in_dst_rank += gs->locals[i / gpu_n]->data[i % gpu_n][server_id * gpu_n + dst_gpu];
                        }
                        temp_intra_redistribute_dst_disp[dst_gpu * gpu_n + src_gpu] = dst_disp_in_dst_rank + cur_step -> restore[prev_src_server][local_rank_id][dst_gpu * gpu_n + src_gpu].offset;
                        local_data_disp += local_data_sz;
                        // if (server_id == 0 && dst_gpu == 0){
                        //     printf("[STEP: %u] RANK %u - REDISTRIBUTE: src rank: %u, dst rank: %u, src disp: %lu, dst disp: %lu, sz: %lu\n", step_id, global_rank_id, cur_src_gpu_global_id, server_id * gpu_n + dst_gpu, temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_dst_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu]);  
                        // }
                    }
                }
                redistribute_sz += reditribute_send_data_sz;
            }
            // expected recv sz
            gs -> flash_sched -> host_stages_intra_redistribute[step_id].recv_sz[dst_gpu] = cur_step -> restore_alltoall_sz[prev_src_server][dst_gpu][local_rank_id];
        }
        plan_chunk_pipeline(&gs -> flash_sched -> host_stages_recv_chunks[step_id], gs -> flash_sched -> host_stages_internode[step_id - 1].recv_sz);
        assign_intra_redistribute_to_blocks(&gs -> flash_sched ->host_stages_intra_redistribute[step_id], temp_intra_redistribute_src_disp, temp_intra_redistribute_dst_disp, temp_intra_redistribute_sz, gpu_n, gs->flash_sched->info.block_n, gs->flash_sched->info.min_tx_sz_at_a_time, gs -> data_size);
        prev_src_server = src_server;
        prev_dst_server = dst_server;
    }

    // final data redistribution
    cur_step = &(gs -> sched -> steps)[ gs -> sched -> step_n - 1];
    memset(&temp_intra_redistribute_src_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
    memset(&temp_intra_redistribute_sz, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
    memset(&temp_intra_redistribute_dst_disp, 0, sizeof(uint64_t) * GPU_NUM_PER_SERVER_SQUARE);
    redistribute_sz = 0;
    for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu++){ // dst gpu id in the current server
        uint64_t reditribute_send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][dst_gpu];
        if (local_rank_id == dst_gpu){
            // get metadata from direct copy
            uint64_t local_data_disp = 0;;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){    // actual data source in the previous server
                // if (global_rank_id == 8) printf("DEBUG1: dst gpu 0, src gpu %u, sz: %lu\n", src_gpu, cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz);
                uint64_t local_data_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
                if (local_data_sz > 0){
                    temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu] = redistribute_sz + local_data_disp;
                    uint cur_src_gpu_global_id = prev_src_server * gpu_n + src_gpu;
                    temp_intra_redistribute_sz[local_rank_id * gpu_n + src_gpu] = local_data_sz;
                    temp_intra_redistribute_dst_disp[local_rank_id * gpu_n + src_gpu] = gs -> flash_buffer_sz_params -> out_disp_per_rank[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset;
                    local_data_disp += local_data_sz;
                    // if (global_rank_id == 0){
                    //     printf("[STEP: %u] RANK %u - DIRECT CPY: src rank: %u, src disp: %lu, dst disp: %lu, sz: %lu\n", gs -> sched -> step_n - 1, global_rank_id, cur_src_gpu_global_id, temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_dst_disp[local_rank_id * gpu_n + src_gpu], temp_intra_redistribute_sz[local_rank_id * gpu_n + src_gpu]);  
                    // }
                }
            }
            redistribute_sz += reditribute_send_data_sz;
        }else{
            uint64_t local_data_disp = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){    // actual data source in the previous server
                uint64_t local_data_sz =  cur_step -> restore[prev_src_server][local_rank_id][dst_gpu * gpu_n + src_gpu].sz;
                if (local_data_sz > 0){
                    temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu] = redistribute_sz + local_data_disp;
                    uint cur_src_gpu_global_id = prev_src_server * gpu_n + src_gpu;
                    temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu] = local_data_sz;
                    uint64_t dst_disp_in_dst_rank = 0;
                    for (uint i = 0; i < cur_src_gpu_global_id; i++){
                        dst_disp_in_dst_rank += gs->locals[i / gpu_n]->data[i % gpu_n][server_id * gpu_n + dst_gpu];
                    }
                    temp_intra_redistribute_dst_disp[dst_gpu * gpu_n + src_gpu] = dst_disp_in_dst_rank + cur_step -> restore[prev_src_server][local_rank_id][dst_gpu * gpu_n + src_gpu].offset;
                    local_data_disp += local_data_sz;
                    // if (server_id == 0 && dst_gpu == 0){
                    //     printf("[STEP: %u] RANK %u - REDISTRIBUTE: src rank: %u, dst rank: %u, src disp: %lu, dst disp: %lu, sz: %lu\n", gs -> sched -> step_n - 1, global_rank_id, cur_src_gpu_global_id, server_id * gpu_n + dst_gpu, temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_dst_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu]);  
                    // }
                }
            }
            redistribute_sz += reditribute_send_data_sz;
        }
        // expected recv sz
        gs -> flash_sched -> host_stages_intra_redistribute[gs -> sched -> step_n - 1].recv_sz[dst_gpu] = cur_step -> restore_alltoall_sz[prev_src_server][dst_gpu][local_rank_id];
    }
    plan_chunk_pipeline(&gs -> flash_sched -> host_stages_recv_chunks[gs -> sched -> step_n - 1], gs -> flash_sched -> host_stages_internode[gs -> sched -> step_n - 2].recv_sz);
    // printf("SEND CHUNK- rank %u => rank %u, first chunk sz: %lu, chunk n: %u, final chunk size: %lu\n", global_rank_id, gs->flash_sched->host_stages_internode[gs -> sched -> step_n - 2].dst_rank , gs -> flash_sched -> host_stages_send_chunks[gs -> sched -> step_n - 2].chunk_sz[0], gs -> flash_sched -> host_stages_send_chunks[gs -> sched -> step_n - 2].chunk_n, gs -> flash_sched -> host_stages_send_chunks[gs -> sched -> step_n - 2].chunk_sz[gs -> flash_sched -> host_stages_send_chunks[gs -> sched -> step_n - 2].chunk_n - 1] );
    // printf("RECV CHUNK- rank %u => rank %u, first chunk sz: %lu, chunk n: %u, final chunk size: %lu\n", gs->flash_sched->host_stages_internode[gs -> sched -> step_n - 2].src_rank, global_rank_id, gs -> flash_sched -> host_stages_recv_chunks[gs -> sched -> step_n - 1].chunk_sz[0], gs -> flash_sched -> host_stages_recv_chunks[gs -> sched -> step_n - 1].chunk_n, gs -> flash_sched -> host_stages_recv_chunks[gs -> sched -> step_n - 1].chunk_sz[gs -> flash_sched -> host_stages_recv_chunks[gs -> sched -> step_n - 1].chunk_n - 1] );


    assign_intra_redistribute_to_blocks(&gs -> flash_sched ->host_stages_intra_redistribute[gs -> sched -> step_n - 1], temp_intra_redistribute_src_disp, temp_intra_redistribute_dst_disp, temp_intra_redistribute_sz, gpu_n, gs->flash_sched->info.block_n, gs->flash_sched->info.min_tx_sz_at_a_time, gs->data_size);
    // if (global_rank_id == 0){
    //     printf("\n=========================================================\n");
    //     for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu++){
    //         printf("dst_gpu %u\n", dst_gpu);
    //         for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
    //             if (temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu] == 0) continue;
    //             printf("\tsrc_gpu: %u\n", src_gpu);
    //             for (uint b = 0; b < gs->flash_sched->info.block_n; b++){
    //                 printf("\t\tblock-%u, src disp: %lu, dst disp: %lu, sz: %lu, rounds: %lu\n", b, gs -> flash_sched -> host_stages_intra_redistribute[gs -> sched -> step_n - 1].src_disp[(dst_gpu * gpu_n + src_gpu) * gs->flash_sched->info.block_n +b], gs -> flash_sched -> host_stages_intra_redistribute[gs -> sched -> step_n - 1].dst_disp[(dst_gpu * gpu_n + src_gpu) * gs->flash_sched->info.block_n +b], gs -> flash_sched -> host_stages_intra_redistribute[gs -> sched -> step_n - 1].sz[(dst_gpu * gpu_n + src_gpu) * gs->flash_sched->info.block_n + b], gs -> flash_sched -> host_stages_intra_redistribute[gs -> sched -> step_n - 1].round_n[(dst_gpu * gpu_n + src_gpu) * gs->flash_sched->info.block_n + b]);
    //             }
    //         }
    //     }
    // }

    // copy metadata to device memory
    CUDACHECK(cudaMemcpy(gs-> flash_sched -> dev_stages_send_chunks, gs -> flash_sched -> host_stages_send_chunks, sizeof(struct chunk_metadata_t) * gs->sched->step_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(gs-> flash_sched -> dev_stages_recv_chunks, gs -> flash_sched -> host_stages_recv_chunks, sizeof(struct chunk_metadata_t) * gs->sched->step_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(gs -> flash_sched -> dev_stages_internode, gs -> flash_sched -> host_stages_internode, sizeof(struct device_inter_p2p_params_t) * gs->sched->step_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(gs -> flash_sched -> dev_stages_intra_redistribute, gs -> flash_sched -> host_stages_intra_redistribute, sizeof(struct device_intra_redistribute_alltoall_params_t) * gs->sched->step_n, cudaMemcpyHostToDevice));

    // if (server_id == 1){
    //     // printf("\n======================OUT BUFFER ========================\n");
    //     // for (uint r = 0; r < rank_n; r++){
    //     //     printf("src gpu: %u, from %lu to %lu, sz: %lu\n", r, gs->flash_buffer_sz_params->out_disp_per_rank[r], gs->flash_buffer_sz_params->out_disp_per_rank[r] + gs->flash_buffer_sz_params->out_sz_per_rank[r], gs->flash_buffer_sz_params->out_sz_per_rank[r]);
    //     // }
    //     uint dst_gpu = 0;
    //     for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
    //         if (temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu] > 0){
    //             printf("\tRDIS - local rank id: %u, src local gpu: %u, src disp: %lu, dst disp: %lu, sz: %lu\n", local_rank_id, src_gpu, temp_intra_redistribute_src_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_dst_disp[dst_gpu * gpu_n + src_gpu], temp_intra_redistribute_sz[dst_gpu * gpu_n + src_gpu]);
    //         }
    //     }
    // }
}

void schedule_this_gpu(struct GlobalScheduler * gs){
    uint global_rank_id = gs->gpu_sched->rankid,
        local_rank_id = gs->gpu_sched->rankid % gs->gpu_sched->gpu_n,
        server_id = gs->gpu_sched->rankid / gs->gpu_sched->gpu_n,
        server_n = gs->gpu_sched->server_n,
        gpu_n = gs->gpu_sched->gpu_n;

    // ------------------------------------------
    // Intrinsic alltoall: sendbuff => recvbuff
    //-------------------------------------------

    for (uint r = 0; r < gpu_n; r++){
        uint cur_gpu = server_id * gpu_n + r;
        uint64_t send_data_sz = gs -> buff_parameter -> sendbuff_sz[cur_gpu];
        if (send_data_sz > 0){
            uint64_t send_data_disp = gs -> buff_parameter -> sendbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].disp = send_data_disp;
            gs -> gpu_sched -> intrinsic_send[gs -> gpu_sched -> intrinsic_send_n].sz = send_data_sz;
            gs -> gpu_sched -> intrinsic_send_n ++;
        }

        uint64_t recv_data_sz = gs -> buff_parameter -> recvbuff_sz[cur_gpu];
        if (recv_data_sz > 0){
            uint64_t recv_data_disp = gs -> buff_parameter -> recvbuff_disp[cur_gpu];
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].gpu = cur_gpu;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> intrinsic_recv[gs -> gpu_sched -> intrinsic_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> intrinsic_recv_n ++;
        }
    }

    // --------------------------------------------------
    // Load balance:
    // First step: lbsend_buff ==> lbrecv_buff
    // Second step: lbrecv_buff ---(memcpy)--> sendbuff
    // --------------------------------------------------

    // first step
    for (uint r = 0; r < gpu_n; r++){
        uint64_t send_data_sz = gs -> buff_parameter -> lbsend_sz[r];
        uint cur_gpu_global_id = server_id * gpu_n + r;
        if (send_data_sz > 0){
            uint64_t send_data_disp = gs -> buff_parameter -> lbsend_disp[r];
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].disp = send_data_disp;
            gs -> gpu_sched -> balance_send[gs -> gpu_sched -> balance_send_n].sz = send_data_sz;
            gs -> gpu_sched -> balance_send_n ++;
        }
        uint64_t recv_data_sz = gs -> buff_parameter -> lbrecv_sz[r];
        if (recv_data_sz > 0){
            uint64_t recv_data_disp = gs -> buff_parameter -> lbrecv_disp[r];
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].gpu = cur_gpu_global_id;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].disp = recv_data_disp;
            gs -> gpu_sched -> balance_recv[gs -> gpu_sched -> balance_recv_n].sz = recv_data_sz;
            gs -> gpu_sched -> balance_recv_n ++;
        }
    }
    // second step
    for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s < server_n; s ++){
                if (s == server_id){
                    continue;
                }
                uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                uint64_t cpy_sz = gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                if (cpy_sz > 0){
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].src_disp =
                        gs -> buff_parameter -> lbrecv_area[src_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].dst_disp =
                        gs -> buff_parameter -> sendbuff_region[dst_gpu_global_id].src_gpu_disp[src_gpu];
                    gs -> gpu_sched -> balance_memcpy[gs -> gpu_sched -> balance_memcpy_n].sz = cpy_sz;
                    gs -> gpu_sched -> balance_memcpy_n ++;
                }
            }
        }
    }


    // ---------------------------------------------------------
    // Cross node: sendbuff ==> crosbuff
    // ---------------------------------------------------------

    //-----------------------------------------------------------
    // Data restore
    // First step: crosbuff ==> restorebuff
    // Second step: crosbuff -- (memcpy) -> recvbuff
    // Second step: restorebuff --(memcpy)-> recvbuff
    //-----------------------------------------------------------

    // calculate buffer size of crosbuff
    uint64_t crosbuff_sz = 0;
    for (uint step_id = 0; step_id < gs->sched->step_n - 1; step_id++){
        for (uint src_svr = 0; src_svr < gs->server_n; src_svr++){
            if (src_svr == server_id) continue;
            if ((gs -> sched -> steps)[step_id].to_server[src_svr] == server_id){
                crosbuff_sz = MAX(crosbuff_sz, (gs -> sched -> steps)[step_id].crossnode_sz[src_svr][local_rank_id]);
                break;
            }
        }
    }
    // make it 512-byte aligned
    crosbuff_sz  = mem_align(crosbuff_sz);

    gs -> buff_parameter->crosbuff_total_sz = crosbuff_sz;
    // gs -> buff_parameter->crosbuff_offset = crosbuff_sz;
    uint64_t rstrbuff_sz = 0, max_rstrbuff_sz = 0;


    // first step
    // uint crosbuff_offset = gs -> buff_parameter -> crosbuff_offset;
    // uint cur_offset = 0, prev_offset = 0;


    uint64_t crossnode_send_disp[MAX_SERVER_NUM];
    memset(crossnode_send_disp, 0 , sizeof(uint64_t) * MAX_SERVER_NUM);
    uint step_id = 0;
    gs -> gpu_sched -> step_n = gs->sched->step_n;
    struct scheduling_step_t * cur_step = &(gs -> sched -> steps)[0];
    struct scheduling_step_gpu_t * cur_gpu_step = &(gs -> gpu_sched -> steps)[0];
    uint dst_server = cur_step -> to_server[server_id];
    uint src_server = cur_step -> from_server[server_id];
    uint dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    uint src_gpu_global_id = src_server * gpu_n + local_rank_id;

    (cur_gpu_step -> crossnode_send).sz = cur_step -> crossnode_sz[server_id][local_rank_id];
    (cur_gpu_step -> crossnode_send).gpu = dst_gpu_global_id;
    (cur_gpu_step -> crossnode_send).disp = gs -> buff_parameter -> sendbuff_disp[dst_server * gpu_n] + crossnode_send_disp[dst_server];
    crossnode_send_disp[dst_server] += (cur_gpu_step -> crossnode_send).sz;

    (cur_gpu_step -> crossnode_recv).sz = cur_step -> crossnode_sz[src_server][local_rank_id];
    (cur_gpu_step -> crossnode_recv).gpu = src_gpu_global_id;
    (cur_gpu_step -> crossnode_recv).disp = 0; // not applying offset here


    // middle steps
    uint prev_dst_server = dst_server,
        prev_src_server = src_server;
    struct scheduling_step_t * prev_step = cur_step;
    struct scheduling_step_gpu_t * prev_gpu_step = cur_gpu_step;
    uint64_t restore_alltoall_senddisp = 0, restore_alltoall_recvdisp = 0, direct_cpy_disp = 0, restore_recvdisp[MAX_GPU_PER_SERVER], direct_cpy_src_disp = 0;

    for (step_id = 1; step_id < gs->sched->step_n - 1; step_id++){
        cur_step = &(gs -> sched -> steps)[step_id];
        cur_gpu_step =  &(gs -> gpu_sched -> steps)[step_id];
        dst_server = cur_step -> to_server[server_id];
        src_server = cur_step -> from_server[server_id];

        dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
        src_gpu_global_id = src_server * gpu_n + local_rank_id;

        // cur_offset = (step_id % 2 == 1) ? crosbuff_offset : 0;
        // prev_offset = (step_id % 2 == 1) ? 0 : crosbuff_offset;
        // cross node transfer

        (cur_gpu_step -> crossnode_send).sz = cur_step -> crossnode_sz[server_id][local_rank_id];
        (cur_gpu_step -> crossnode_send).gpu = dst_gpu_global_id;
        (cur_gpu_step -> crossnode_send).disp = gs -> buff_parameter -> sendbuff_disp[dst_server * gpu_n] + crossnode_send_disp[dst_server];
        crossnode_send_disp[dst_server] += (cur_gpu_step -> crossnode_send).sz;

        (cur_gpu_step -> crossnode_recv).sz = cur_step -> crossnode_sz[src_server][local_rank_id];
        (cur_gpu_step -> crossnode_recv).gpu = src_gpu_global_id;
        (cur_gpu_step -> crossnode_recv).disp = 0; // applying offset here

        //restore data from previous step

        // restore alltoall
        restore_alltoall_senddisp = 0;
        restore_alltoall_recvdisp = 0;
        direct_cpy_disp = 0;
        rstrbuff_sz = 0;
        memset(restore_recvdisp, 0, sizeof(uint64_t) * MAX_GPU_PER_SERVER);
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            uint64_t send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][local_gpu];
            if (local_gpu == local_rank_id){
                direct_cpy_disp = restore_alltoall_senddisp;
                restore_alltoall_senddisp += send_data_sz;
                continue;
            }

            uint cur_gpu = server_id * gpu_n + local_gpu;
            if (send_data_sz > 0){
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].gpu = cur_gpu;
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].disp = restore_alltoall_senddisp;
                cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].sz = send_data_sz;
                restore_alltoall_senddisp += send_data_sz;
                cur_gpu_step -> restore_send_n ++;
            }
            uint64_t recv_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_gpu][local_rank_id];
            if (recv_data_sz > 0){
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].gpu = cur_gpu;
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].disp = restore_alltoall_recvdisp;
                restore_recvdisp[local_gpu] = restore_alltoall_recvdisp;
                cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].sz = recv_data_sz;
                restore_alltoall_recvdisp += recv_data_sz;
                cur_gpu_step -> restore_recv_n ++;
                // calculate restore buffer size
                rstrbuff_sz += recv_data_sz;
            }
        }
        max_rstrbuff_sz = MAX(max_rstrbuff_sz, rstrbuff_sz);

        // direct copy
        direct_cpy_src_disp = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint64_t cpy_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
            if (cpy_sz > 0){
                uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].sz = cpy_sz;
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].src_disp = direct_cpy_disp + direct_cpy_src_disp;
                cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset ;
                direct_cpy_src_disp += cpy_sz;
                cur_gpu_step -> direct_memcpy_n ++;
            }
        }

        // restore memcpy
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            uint64_t restore_cpy_src_disp = 0;
            for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
                uint64_t cpy_sz = cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].sz;
                if(cpy_sz > 0){
                    uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].sz = cpy_sz;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].src_disp = restore_recvdisp[local_gpu] + restore_cpy_src_disp;
                    cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].offset;
                    restore_cpy_src_disp += cpy_sz;
                    cur_gpu_step -> restore_memcpy_n ++;
                }
            }
        }

        prev_src_server = src_server;
        prev_dst_server = dst_server;
    }

    // final restore
    // prev_offset = ((gs -> sched -> step_n - 1) % 2 == 1) ? 0 : crosbuff_offset;
    cur_step = &(gs -> sched -> steps)[ gs -> sched -> step_n - 1];
    cur_gpu_step = &(gs -> gpu_sched -> steps)[ gs -> gpu_sched -> step_n - 1];

    // restore alltoall
    restore_alltoall_senddisp = 0;
    restore_alltoall_recvdisp = 0;
    direct_cpy_disp = 0;
    rstrbuff_sz = 0;
    memset(restore_recvdisp, 0, sizeof(uint64_t) * MAX_GPU_PER_SERVER);
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        uint64_t send_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_rank_id][local_gpu];
        if (local_gpu == local_rank_id){
            direct_cpy_disp = restore_alltoall_senddisp;
            restore_alltoall_senddisp += send_data_sz;
            continue;
        }

        uint cur_gpu = server_id * gpu_n + local_gpu;
        if (send_data_sz > 0){
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].gpu = cur_gpu;
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].disp = restore_alltoall_senddisp;
            cur_gpu_step -> restore_send[ cur_gpu_step -> restore_send_n].sz = send_data_sz;
            restore_alltoall_senddisp += send_data_sz;
            cur_gpu_step -> restore_send_n ++;
        }
        uint64_t recv_data_sz = cur_step -> restore_alltoall_sz[prev_src_server][local_gpu][local_rank_id];
        if (recv_data_sz > 0){
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].gpu = cur_gpu;
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].disp = restore_alltoall_recvdisp;
            restore_recvdisp[local_gpu] = restore_alltoall_recvdisp;
            cur_gpu_step -> restore_recv[ cur_gpu_step -> restore_recv_n].sz = recv_data_sz;
            restore_alltoall_recvdisp += recv_data_sz;
            cur_gpu_step -> restore_recv_n ++;
            // calculate restore buffer size
            rstrbuff_sz += recv_data_sz;
        }
    }
    max_rstrbuff_sz = MAX(max_rstrbuff_sz, rstrbuff_sz);

    // direct copy
    direct_cpy_src_disp = 0;
    for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
        uint64_t cpy_sz = cur_step->direct_cpy[prev_src_server][local_rank_id][src_gpu].sz;
        if (cpy_sz > 0){
            uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].sz = cpy_sz;
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].src_disp = direct_cpy_disp + direct_cpy_src_disp;
            cur_gpu_step -> direct_memcpy[cur_gpu_step -> direct_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> direct_cpy[prev_src_server][local_rank_id][src_gpu].offset ;
            direct_cpy_src_disp += cpy_sz;
            cur_gpu_step -> direct_memcpy_n ++;
        }
    }

    // restore memcpy
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        uint64_t restore_cpy_src_disp = 0;
        for (uint src_gpu = 0; src_gpu < gpu_n; src_gpu ++){
            uint64_t cpy_sz = cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].sz;
            if(cpy_sz > 0){
                uint cur_src_gpu_global_id = prev_src_server  * gpu_n + src_gpu;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].sz = cpy_sz;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].src_disp = restore_recvdisp[local_gpu] + restore_cpy_src_disp;
                cur_gpu_step -> restore_memcpy[cur_gpu_step -> restore_memcpy_n].dst_disp = gs -> buff_parameter -> recvbuff_disp[cur_src_gpu_global_id] + cur_step -> restore[prev_src_server][local_gpu][local_rank_id * gpu_n + src_gpu].offset;
                restore_cpy_src_disp += cpy_sz;
                cur_gpu_step -> restore_memcpy_n ++;
            }
        }
    }
    // make it 512-byte-aligned
    gs -> buff_parameter -> rstrbuff_total_sz = mem_align(max_rstrbuff_sz);
}
