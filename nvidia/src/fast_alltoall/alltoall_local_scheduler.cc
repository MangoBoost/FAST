#include "fast_alltoall/alltoall_local_scheduler.h"
#include <iomanip>
#include <iostream>
#include <stdio.h>
#if ROCM_RCCL_COMPILE
#include <hip/hip_runtime.h>
#endif
#if CUDA_NCCL_COMPILE
#include <cuda_runtime.h>
#endif



void init_local_scheduler(struct LocalScheduler * ls, uint64_t* _data, uint _gpu_n, uint _server_n, uint _server_id){
    ls->gpu_n = _gpu_n;
    ls->server_n = _server_n;
    ls->server_id = _server_id;
    uint dim = ls->gpu_n * ls->server_n;
    // hipMallocManaged((void**) &ls->data, sizeof(uint*) * ls->gpu_n);
    // hipMallocManaged((void**) &ls->balanced_data, sizeof(uint*) * ls->gpu_n);
    // hipMallocManaged((void**) &ls->data_after_balance, sizeof(data_t*) * ls->gpu_n);
    ls->data = (uint64_t**) malloc(sizeof(uint64_t *) * ls->gpu_n);
    ls->balanced_data = (uint64_t **) malloc(sizeof(uint64_t*) * ls->gpu_n);
    ls->data_after_balance = (data_t **) malloc(sizeof(data_t*) * ls->gpu_n);
    // data = new uint*[gpu_n];
    // balanced_data = new uint*[gpu_n];
    // data_after_balance = new data_t*[gpu_n];
    uint idx = 0;
    for (uint i = 0; i < ls->gpu_n; i++){
        // hipMallocManaged((void**)&ls->data[i], sizeof(uint) * dim);
        // hipMallocManaged((void**)&ls->balanced_data[i], sizeof(uint) * dim);
        // hipMallocManaged((void**)&ls->data_after_balance[i], sizeof(data_t) * dim);
        ls->data[i] = (uint64_t*) malloc(sizeof(uint64_t) * dim);
        ls->balanced_data[i] = (uint64_t*) malloc(sizeof(uint64_t) * dim);
        ls->data_after_balance[i] = (data_t*) malloc(sizeof(data_t) * dim);
        // data[i] = new uint[dim];
        // balanced_data[i] = new uint[dim];
        // data_after_balance[i] = new data_t[dim];
        for (uint j = 0; j < dim; j++){
            ls->data[i][j] = _data[idx];
            ls->balanced_data[i][j] = _data[idx];
            for (uint z = 0; z < MAX_GPU_PER_SERVER; z++){
                ls->data_after_balance[i][j].sz[z] = 0;
                ls->data_after_balance[i][j].offset[z] = 0;
            }
            ls->data_after_balance[i][j].sz[i] = _data[idx];
            ls->data_after_balance[i][j].sum = _data[idx];
            idx++;
        }
    }
    ls->server2server_data = (uint64_t *) malloc(sizeof(uint64_t) * ls->server_n);
    memset(ls->server2server_data, 0, ls->server_n * sizeof(uint64_t));
    ls->row_sum = (uint64_t *) malloc(sizeof(uint64_t) * ls->gpu_n * ls->server_n);
    memset(ls->row_sum, 0, ls->gpu_n * ls->server_n * sizeof(uint64_t));
    ls->intrinsic_all2all = (uint64_t *) malloc( sizeof(uint64_t) *  ls->gpu_n * ls->gpu_n);
    memset(ls->intrinsic_all2all, 0, ls->gpu_n * ls->gpu_n * sizeof(uint64_t));

    // hipMallocManaged((void**)&ls->server2server_data, sizeof(uint) * ls->server_n);
    // hipMemset(ls->server2server_data, 0, ls->server_n * sizeof(uint));
    // hipMallocManaged((void**)&ls->row_sum, sizeof(uint) * ls->gpu_n * ls->server_n);
    // hipMemset(ls->row_sum, 0, ls->gpu_n * ls->server_n * sizeof(uint));
    // hipMallocManaged((void**)&ls->intrinsic_all2all, sizeof(uint) * ls->gpu_n * ls->gpu_n);
    // hipMemset(ls->intrinsic_all2all, 0, ls->gpu_n * ls->gpu_n * sizeof(uint));
    // server2server_data = new uint[server_n];
    // memset(server2server_data, 0, server_n * sizeof(uint));
    // row_sum = new uint[gpu_n * server_n];
    // memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    // intrinsic_all2all = new uint[gpu_n * gpu_n];
    // memset(intrinsic_all2all, 0, gpu_n * gpu_n * sizeof(uint));
    prepare_load_balance(ls);
}

void update_local_scheduler(struct LocalScheduler * ls, uint64_t* _data){
    uint dim = ls->gpu_n * ls->server_n;
    uint idx = 0;
    for (uint i = 0; i < ls->gpu_n; i++){
        for (uint j = 0; j < dim; j++){
            ls->data[i][j] = _data[idx];
            ls->balanced_data[i][j] = _data[idx];
            for (uint z = 0; z < MAX_GPU_PER_SERVER; z++){
                ls->data_after_balance[i][j].sz[z] = 0;
                ls->data_after_balance[i][j].offset[z] = 0;
            }
            ls->data_after_balance[i][j].sz[i] = _data[idx];
            ls->data_after_balance[i][j].sum = _data[idx];
            idx++;
        }
    }
    memset(ls->server2server_data, 0, ls->server_n * sizeof(uint64_t));
    memset(ls->row_sum, 0, ls->gpu_n * ls->server_n * sizeof(uint64_t));
    memset(ls->intrinsic_all2all, 0, ls->gpu_n * ls->gpu_n * sizeof(uint64_t));
    prepare_load_balance(ls);
}

void free_local_scheduler(struct LocalScheduler * ls){
    for (uint i = 0; i < ls->gpu_n; i++){
        free(ls->data[i]);
        free(ls->balanced_data[i]);
        free(ls->data_after_balance[i]);

        // hipFree(ls->data[i]);
        // hipFree(ls->balanced_data[i]);
        // hipFree(ls->data_after_balance[i]);
        // delete[] data[i];
        // delete[] balanced_data[i];
        // delete[] data_after_balance[i];
    }
    free(ls->data);
    free(ls->balanced_data);
    free(ls->data_after_balance);
    free(ls->server2server_data);
    free(ls->row_sum);
    free(ls->intrinsic_all2all);

    // hipFree(ls->data);
    // hipFree(ls->balanced_data);
    // hipFree(ls->data_after_balance);
    // hipFree(ls->server2server_data);
    // hipFree(ls->row_sum);
    // hipFree(ls->intrinsic_all2all);
    // delete[] data;
    // delete[] balanced_data;
    // delete[] data_after_balance;
    // delete[] server2server_data;
    // delete[] row_sum;
    // delete[] intrinsic_all2all;
}

void prepare_load_balance(struct LocalScheduler * ls){
    memset(ls->row_sum, 0, ls->gpu_n * ls->server_n * sizeof(uint64_t));
    // hipMemset(ls->row_sum, 0, ls->gpu_n * ls->server_n * sizeof(uint));
    // memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    for (uint i = 0; i < ls->server_n; i++){
        if (i == ls->server_id){
            ls->server2server_data[i] = 0;
            for (uint j = 0; j < ls->gpu_n; j++){
                for (uint k = 0; k < ls->gpu_n; k++){
                    ls->intrinsic_all2all[j * ls->gpu_n + k] = ls->data[j][ls->server_id * ls->gpu_n + k];
                }
            }
            continue;
        }
        for (uint j = 0; j < ls->gpu_n; j++){
            // for each row at each tile
            ls->row_sum[i * ls->gpu_n + j] = 0;
            for (uint k = 0; k < ls->gpu_n; k++){
                ls->row_sum[i * ls->gpu_n + j] += ls->data[j][i * ls->gpu_n + k];
            }
        }

        uint64_t row_avg = 0;
        for (uint k = 0; k < ls->gpu_n; k++){
            row_avg += ls->row_sum[i * ls->gpu_n + k];
        }
        ls->server2server_data[i] = (row_avg + ls->gpu_n - 1) / ls->gpu_n;
    }
    // print();
    // for (uint i = 0; i < server_n; i++){
    //     cout << "to server " << i << ":" << server2server_data[i] << endl;
    // }
}


void balance_one_server(struct LocalScheduler * ls, uint to_server_id, struct balance_data_t (*r)[MAX_GPU_PER_SERVER_SQUARE]){
    if (to_server_id == ls->server_id){
        return;
    }

    uint smaller_row[MAX_GPU_PER_SERVER], smaller_row_n = 0;
    uint bigger_row[MAX_GPU_PER_SERVER], bigger_row_n = 0;

    for (uint i = 0; i < ls->gpu_n; i++){
        if (ls->row_sum[to_server_id * ls->gpu_n + i] < ls->server2server_data[to_server_id]){
            smaller_row[smaller_row_n] = i;
            smaller_row_n ++;
        }else if (ls->row_sum[to_server_id * ls->gpu_n + i] > ls->server2server_data[to_server_id]){
            bigger_row[bigger_row_n] = i;
            bigger_row_n ++;
        }
    }

    for (uint big_row_id = 0; big_row_id < bigger_row_n; big_row_id++){
        uint big_row = bigger_row[big_row_id];

        int64_t rm_data = ls->row_sum[to_server_id * ls->gpu_n + big_row] - ls->server2server_data[to_server_id];
        for (uint small_row_id = 0; small_row_id < smaller_row_n; small_row_id++){
            uint small_row = smaller_row[small_row_id];
            if (ls->row_sum[to_server_id * ls->gpu_n + small_row] == ls->server2server_data[to_server_id]){
                continue;
            }

            for (uint j = 0; j < ls->gpu_n; j++){
                // check each element of the big row
                int64_t mv_data = MIN(MIN(rm_data, ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].sum), ls->server2server_data[to_server_id] - ls->row_sum[to_server_id * ls->gpu_n + small_row]);
                // cout << "LB scheduler, mv data: " << mv_data <<", channel: "<< j << ", src gpu: " << *big_row << ", dst gpu: " << *small_row << endl;

                if (mv_data == 0){
                    continue;
                }
                rm_data -= mv_data;
                // big row col j ====> small row col j via balance big row -> small row
                ls->row_sum[to_server_id * ls->gpu_n + small_row] += mv_data;
                ls->row_sum[to_server_id * ls->gpu_n + big_row] -= mv_data;
                (*r)[big_row * ls->gpu_n + small_row].sz[j] += mv_data;
                (*r)[big_row * ls->gpu_n + small_row].offset[j] = ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].offset[big_row];

                // cout << "lb server" << server_id << ", dst server" << to_server_id << ", mv data: " << mv_data<<", big row: " << *big_row << ", small row: " << *small_row <<", lb dst gpu: " << j << endl;
                ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].sz[big_row] -= mv_data;
                ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].sum -= mv_data;
                ls->data_after_balance[small_row][to_server_id * ls->gpu_n + j].offset[big_row] = ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].offset[big_row];
                ls->data_after_balance[big_row][to_server_id * ls->gpu_n + j].offset[big_row] += mv_data;
                ls->data_after_balance[small_row][to_server_id * ls->gpu_n + j].sz[big_row] += mv_data;
                ls->data_after_balance[small_row][to_server_id * ls->gpu_n + j].sum += mv_data;
                if (rm_data == 0){
                    break;
                }
            }
            if (rm_data == 0){
                break;
            }
        }
    }

}


void restore_one_server(struct LocalScheduler * ls, uint to_server_id, uint64_t (*channel)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t (*crossnode_sz)[MAX_GPU_PER_SERVER], struct recv_data_t (*r)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t (*restore_alltoall_sz)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER], struct recv_data_t (*dcpy)[MAX_GPU_PER_SERVER][MAX_GPU_PER_SERVER_SQUARE], uint64_t freq){
    if (to_server_id == ls->server_id){
        return;
    }
    // uint * row_transfer = new uint[gpu_n];
    uint64_t * row_transfer = (uint64_t *) malloc (sizeof(uint64_t) * ls->gpu_n);
    // hipMallocManaged((void**) &row_transfer, sizeof(uint) * ls->gpu_n);
    for (uint i = 0; i < ls->gpu_n; i++){    // src gpu

        row_transfer[i] = 0;
        for (uint j = 0; j < ls->gpu_n; j++){   // dst gpu
            (*restore_alltoall_sz)[i][j] = 0;
            for (uint from_gpu = 0; from_gpu < ls->gpu_n; from_gpu++){
                if (ls->data_after_balance[i][to_server_id * ls->gpu_n + j].sz[from_gpu] == 0){
                    continue;
                }
                int64_t transfer = MIN(freq - row_transfer[i], ls->data_after_balance[i][to_server_id * ls->gpu_n + j].sz[from_gpu]);
                row_transfer[i] += transfer;
                (*restore_alltoall_sz)[i][j] += transfer;
                (*channel)[i][ j * ls->gpu_n + from_gpu] += transfer;
                ls->data_after_balance[i][to_server_id * ls->gpu_n + j].sz[from_gpu] -= transfer;
                ls->row_sum[to_server_id * ls->gpu_n + i] -= transfer;
                if (i != j){    // Transfer is Server m's GPUi --> Server n's GPUi, need to dispatch data from m's GPUi --> n's GPUj if i not equal j
                    //     cout << "restore server" << server_id << ", dst server" << to_server_id << ", restore data: " << transfer<<", src gpu: " << i << ", dst gpu: " << j << endl;
                    // cout << "src : "<< i << ", dst: " << j << ", from gpu: " << from_gpu <<", size: " << transfer << endl;
                    (*r)[i][j * ls->gpu_n + from_gpu].sz += transfer;
                    (*r)[i][j * ls->gpu_n + from_gpu].offset = ls->data_after_balance[i][to_server_id * ls->gpu_n + j].offset[from_gpu];
                }else{
                    (*dcpy)[i][from_gpu].sz += transfer;
                    (*dcpy)[i][from_gpu].offset = ls->data_after_balance[i][to_server_id * ls->gpu_n + j].offset[from_gpu];
                }
                ls->data_after_balance[i][to_server_id * ls->gpu_n + j].offset[from_gpu] += transfer;
                if (row_transfer[i] == freq){
                    break;
                }
            }
            if (row_transfer[i] == freq){
                break;
            }

        }
        (*crossnode_sz)[i] = row_transfer[i];

    }
    free(row_transfer);
    // hipFree(row_transfer);
    // delete[] row_transfer;

}




void print_local_scheduler(struct LocalScheduler * ls, uint dst_server_id){
    // cout << "server "<< ls->server_id << " to server " << dst_server_id << endl;
    for (uint i = 0; i < ls->gpu_n; i++){
        for (uint j = 0; j < ls->gpu_n; j++){
            FLASHLOG("%*lu", 10,  ls->data[i][dst_server_id * ls->gpu_n + j]);
        }
        FLASHLOG("\n");
    }
}

void print_local_scheduler(struct LocalScheduler * ls){
    uint dim = ls->gpu_n * ls->server_n;

    // cout << "original matrix: " << endl;
    for (uint i = 0; i < ls->gpu_n; i++){
        for (uint j = 0; j < dim; j++){
            FLASHLOG("%*lu", 10,  ls->data[i][j]);
        }
        FLASHLOG("\n");
    }

    // cout << endl << "balanced matrix: " << endl;
    // for (uint i = 0; i < gpu_n; i++){
    //     for (uint j = 0; j < dim; j++){
    //         cout << setw(10);
    //         cout << balanced_data[i][j];
    //     }
    //     cout << endl;
    // }

    // cout << endl << "----------------" << endl << "Intra All2All" << endl;
    // for (auto it = intra_all2all.begin(); it != intra_all2all.end(); it++){
    //     cout << "server " << server_id << " to server " << it->first << endl;
    //     cout << "balance:" << endl;
    //     for (uint i = 0; i < gpu_n; i++){
    //         for (uint j = 0; j < gpu_n; j++){
    //             cout << setw(10);
    //             cout << it->second.balance[i*gpu_n + j];
    //         }
    //         cout << endl;
    //     }
    //     cout << endl<< "dispatch:" << endl;
    //     for (uint i = 0; i < gpu_n; i++){
    //         for (uint j = 0; j < gpu_n; j++){
    //             cout << setw(10);
    //             cout << it->second.dispatch[i*gpu_n + j];
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }
}

void print_matrix(uint64_t * data, uint m, uint n){ //width m, height n
    std::cout<<"--------------------------------------" << std::endl;
    for(uint i = 0; i < n; i ++){
        for (uint j = 0; j < m; j++){
            std::cout << std::setw(10);
            std::cout << data[i * m + j] << ",";
        }
        std::cout<<std::endl;
    }
    std::cout << "--------------------------------------" << std::endl;
}