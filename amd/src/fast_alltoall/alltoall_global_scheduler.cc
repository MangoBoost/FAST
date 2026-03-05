#include "fast_alltoall/alltoall_global_scheduler.h"
#include "fast_alltoall/alltoall_define.h"
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>


void init_global_scheduler(struct GlobalScheduler * gs, uint _server_n, uint _gpu_n, uint64_t * demand_matrix,  uint rankid){
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

    memset(gs -> buff_parameter, 0 , sizeof(buffer_parameter_t));

    gs -> opt.algbw_limit = 0;
    gs -> opt.crossnode_time_limit = 0;
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
    // hipFree(gs->sched);
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
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint));
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


    uint64_t workload_sz = 0;
    uint64_t max_cross_node_sz = 0;
    for (uint i = 0; i < gs->server_n; i++){
        uint64_t cross_node_sz = 0;
        for (uint j = 0; j < gs->gpu_n; j++){
            for (uint z = 0; z < gs->gpu_n * gs->server_n; z++){
                workload_sz += gs->locals[i]->data[j][z];
                if (i != z / gs->gpu_n){
                    cross_node_sz += gs->locals[i]->data[j][z];
                }
            }
        }
        max_cross_node_sz = MAX(max_cross_node_sz, cross_node_sz);

    }
    gs -> opt.crossnode_time_limit = max_cross_node_sz * sizeof(int32_t) * 1e-6 / gs->gpu_n / (double) ROCEV2_PAYLOAD_TPUT;
    gs -> opt.algbw_limit = workload_sz * sizeof(int32_t) * 1e-6 / (gs->gpu_n * gs->server_n) / gs -> opt.crossnode_time_limit;
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
        memcpy(gs->sched->intrinsic_ata[src_svr], gs->locals[lid]->intrinsic_all2all, gs->gpu_n * gs->gpu_n * sizeof(uint));
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
