#include "local.h"
#include <vector>
#include <iomanip>
#include <stdio.h>


using namespace std;

LocalScheduler::LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id, INTRA_LINK_TYPE _type){
    gpu_n = _gpu_n;
    server_n = _server_n;
    server_id = _server_id;
    uint dim = gpu_n * server_n;
    // print_matrix(_data, dim, gpu_n);
    data = new uint*[gpu_n];
    balanced_data = new uint*[gpu_n];
    uint idx = 0;
    for (uint i = 0; i < gpu_n; i++){
        data[i] = new uint[dim];
        balanced_data[i] = new uint[dim];
        for (uint j = 0; j < dim; j++){
            data[i][j] = _data[idx];
            balanced_data[i][j] = _data[idx];
            idx++;
        }
    }
    intra_type = _type;
    intra_info = get_intra_link_info(intra_type);
    server2server_data = new uint[server_n];
    memset(server2server_data, 0, server_n * sizeof(uint));
    row_sum = new uint[gpu_n * server_n];
    memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    intrinsic_all2all = new uint[gpu_n * gpu_n];
    memset(intrinsic_all2all, 0, gpu_n * gpu_n * sizeof(uint));
}

LocalScheduler::LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id, struct link_info_t _intra_link){
    gpu_n = _gpu_n;
    server_n = _server_n;
    server_id = _server_id;
    uint dim = gpu_n * server_n;
    // print_matrix(_data, dim, gpu_n);
    data = new uint*[gpu_n];
    balanced_data = new uint*[gpu_n];
    uint idx = 0;
    for (uint i = 0; i < gpu_n; i++){
        data[i] = new uint[dim];
        balanced_data[i] = new uint[dim];
        for (uint j = 0; j < dim; j++){
            data[i][j] = _data[idx];
            balanced_data[i][j] = _data[idx];
            idx++;
        }
    }
    intra_info = _intra_link;
    server2server_data = new uint[server_n];
    memset(server2server_data, 0, server_n * sizeof(uint));
    row_sum = new uint[gpu_n * server_n];
    memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    intrinsic_all2all = new uint[gpu_n * gpu_n];
    memset(intrinsic_all2all, 0, gpu_n * gpu_n * sizeof(uint));
}

LocalScheduler::~LocalScheduler(){
    for (uint i = 0; i < gpu_n; i++){
        delete[] data[i];
    }
    delete[] data;
    delete[] balanced_data;
    delete[] server2server_data;
    for (auto it = intra_all2all.begin(); it !=intra_all2all.end(); it++){
        struct load_balance_result r = it->second;
        delete[] r.balance;
        delete[] r.dispatch;
    }
    delete[] row_sum;
    delete[] intrinsic_all2all;
}

void LocalScheduler::load_balance(){
    for (auto it = intra_all2all.begin(); it !=intra_all2all.end(); it++){
        struct load_balance_result r = it->second;
        delete[] r.balance;
        delete[] r.dispatch;
    }

    struct load_balance_result r;
    //initialize data space for intra-all2all
    r.balance = new uint[gpu_n * gpu_n];
    r.dispatch = new uint[gpu_n * gpu_n];
    memset(r.balance, 0, gpu_n * gpu_n * sizeof(uint));
    memset(r.dispatch, 0, gpu_n * gpu_n * sizeof(uint));
    prepare_load_balance();

    print();
    // for (uint i = 0; i < 3; i++){
    //     server2server_balance(1, r, 6 - i);
    //     print_load_balance_step(&r);
    //     cout << "--------------------------" << endl;
    //     print();
    //     memset(r.balance, 0, gpu_n * gpu_n * sizeof(uint));
    //     memset(r.dispatch, 0, gpu_n * gpu_n * sizeof(uint));
    // }

    for (uint i = 0; i < server_n; i++){
        if (i == server_id)
            continue;
        server2server_balance(i, r);
        print_load_balance_step(&r);
        cout << "--------------------------" << endl;
        print();
        memset(r.balance, 0, gpu_n * gpu_n * sizeof(uint));
        memset(r.dispatch, 0, gpu_n * gpu_n * sizeof(uint));
    }

    cout << "dispatch: " << endl;
    for (uint i = 0; i < 3; i++){
        server2server_dispatch(1, r, 6 - i);
        print_load_balance_step(&r);
        cout << "--------------------------" << endl;
        print();
        memset(r.balance, 0, gpu_n * gpu_n * sizeof(uint));
        memset(r.dispatch, 0, gpu_n * gpu_n * sizeof(uint));
    }



    // for (uint i = 0; i < server_n; i++){
    //     if (i == server_id){
    //         continue;
    //     }
    //     struct load_balance_result r;
    //     //initialize data space for intra-all2all
    //     r.balance = new uint[gpu_n * gpu_n];
    //     memset(r.balance, 0, gpu_n * gpu_n * sizeof(uint));
    //     r.dispatch = new uint[gpu_n * gpu_n];
    //     memset(r.dispatch, 0, gpu_n * gpu_n * sizeof(uint));
    //     server2server_balance(i, &r, &server2server_data[i]);
    //     intra_all2all.insert(make_pair(i, r));
    // }
    // server2server_data[server_id] = 0;

    delete[] r.balance;
    delete[] r.dispatch;
}

void LocalScheduler::prepare_load_balance(){
    memset(row_sum, 0, gpu_n * server_n * sizeof(uint));
    for (uint i = 0; i < server_n; i++){
        if (i == server_id){
            server2server_data[i] = 0;
            for (uint j = 0; j < gpu_n; j++){
                for (uint k = 0; k < gpu_n; k++){
                    intrinsic_all2all[j * gpu_n + k] = data[j][server_id * gpu_n + k];
                }
            }
            continue;
        }
        for (uint j = 0; j < gpu_n; j++){
            // for each row at each tile
            row_sum[i * gpu_n + j] = 0;
            for (uint k = 0; k < gpu_n; k++){
                row_sum[i * gpu_n + j] += data[j][i * gpu_n + k];
            }
        }

        uint row_avg = 0;
        for (uint k = 0; k < gpu_n; k++){
            row_avg += row_sum[i * gpu_n + k];
        }
        server2server_data[i] = (row_avg + gpu_n - 1) / gpu_n;
    }
    // print();
    // for (uint i = 0; i < server_n; i++){
    //     cout << "to server " << i << ":" << server2server_data[i] << endl;
    // }
}

void LocalScheduler::server2server_approx(uint to_server_id){
    uint max_row = 0;
    for (uint k = 0; k < gpu_n; k++){
        max_row = MAX(row_sum[to_server_id * gpu_n + k], max_row);
    }
    server2server_data[to_server_id] = max_row;
}

bool LocalScheduler::check_balance_dispatch(uint to_server_id, double MBpu){
    double beta_cost = 0, alpha_cost = 0;
    // raw balance cost
    uint max_row = 0;
    for (uint k = 0; k < gpu_n; k++){
        max_row = MAX(row_sum[to_server_id * gpu_n + k], max_row);
    }
    alpha_cost += intra_info.alpha;
    beta_cost += max_row * intra_info.beta * MBpu;

    // raw dispatch cost
    int largest_in_diagnal = 0;
    for (uint i = 0; i < gpu_n; i ++){
        largest_in_diagnal = 0;
        for (uint j = 0; j < gpu_n; j++){
            largest_in_diagnal = MAX(largest_in_diagnal, balanced_data[j][to_server_id * gpu_n + ((j + i) % gpu_n)]);
        }
        if (largest_in_diagnal != 0){
            alpha_cost += intra_info.alpha;
            beta_cost  += largest_in_diagnal * intra_info.beta * MBpu;
        }
    }
    // cout << "beta cost: " << beta_cost << ", alpha cost: " << alpha_cost << ", ratio: " << beta_cost / alpha_cost << endl;
    
    return (beta_cost / alpha_cost) < 2;

    // baseline: (m*n - 1) steps inter-server
    // initial fastall2all: m*n steps inter-server, each m steps intra-server
    // after optimization fastall2all: n steps inter-server, each m steps intra-server
}


void LocalScheduler::server2server_balance(uint to_server_id, struct load_balance_result r, uint freq){
    vector<uint> smaller_row;
    vector<uint> bigger_row;
    for (uint i = 0; i < gpu_n; i++){
        if (row_sum[to_server_id * gpu_n + i] < freq){
            smaller_row.push_back(i);
        }else if (row_sum[to_server_id * gpu_n + i] > freq){
            bigger_row.push_back(i);
        }
    }

    for (auto small_row = smaller_row.begin(); small_row != smaller_row.end(); small_row ++){
        uint offset = freq - row_sum[to_server_id * gpu_n + *small_row];

        for (auto big_row = bigger_row.begin(); big_row != bigger_row.end(); ){

            for (uint j = 0; j < gpu_n; j++){
                uint move_data = MIN(row_sum[to_server_id * gpu_n + *big_row] - freq, MIN(offset, balanced_data[*big_row][to_server_id * gpu_n + j]));
                offset -= move_data;
                // big row -> j ====> small row -> j via balance big row -> small row
                row_sum[to_server_id * gpu_n + *big_row] -= move_data;
                row_sum[to_server_id * gpu_n + *small_row] += move_data;
                balanced_data[*big_row][to_server_id * gpu_n + j] -= move_data;
                balanced_data[*small_row][to_server_id * gpu_n + j] += move_data;
                r.balance[ (*big_row) * gpu_n + (*small_row)] += move_data;
            }

            if (offset == 0){
                break;
            }

            if (row_sum[to_server_id * gpu_n + *big_row] == freq){
                big_row = bigger_row.erase(big_row);
            }else{
                big_row ++;
            }

        }
        if (bigger_row.empty()){
            break;
        }
    }

    uint * row_transfer = new uint[gpu_n];

    for (uint i = 0; i < gpu_n; i++){

        row_transfer[i] = 0;
        for (uint j = 0; j < gpu_n; j++){
            uint transfer = MIN(freq - row_transfer[i], balanced_data[i][to_server_id * gpu_n + j]);
            row_transfer[i] += transfer;
            balanced_data[i][to_server_id * gpu_n + j] -= transfer;
            row_sum[to_server_id * gpu_n + i] -= transfer;
            if (i != j){    // Transfer is Server m's GPUi --> Server n's GPUi, need to dispatch data from m's GPUi --> n's GPUj if i not equal j
                r.dispatch[i * gpu_n + j] += transfer;
            }
            if (row_transfer[i] == freq){
                break;
            }
        }

    }
    delete[] row_transfer;
}

void LocalScheduler::server2server_dispatch(uint to_server_id, struct load_balance_result r, uint freq){
    uint * row_transfer = new uint[gpu_n];

    for (uint i = 0; i < gpu_n; i++){

        row_transfer[i] = 0;
        for (uint j = 0; j < gpu_n; j++){
            uint transfer = MIN(freq - row_transfer[i], balanced_data[i][to_server_id * gpu_n + j]);
            row_transfer[i] += transfer;
            balanced_data[i][to_server_id * gpu_n + j] -= transfer;
            row_sum[to_server_id * gpu_n + i] -= transfer;
            if (i != j){    // Transfer is Server m's GPUi --> Server n's GPUi, need to dispatch data from m's GPUi --> n's GPUj if i not equal j
                r.dispatch[i * gpu_n + j] += transfer;
            }
            if (row_transfer[i] == freq){
                break;
            }
        }

    }
    delete[] row_transfer;
}


void LocalScheduler::server2server_balance(uint to_server_id, struct load_balance_result r){

    vector<uint> smaller_row;
    vector<uint> bigger_row;

    for (uint i = 0; i < gpu_n; i++){
        if (row_sum[to_server_id * gpu_n + i] < server2server_data[to_server_id]){
            smaller_row.push_back(i);
        }else if (row_sum[to_server_id * gpu_n + i] > server2server_data[to_server_id]){
            bigger_row.push_back(i);
        }
    }

    for (auto big_row = bigger_row.begin(); big_row != bigger_row.end(); big_row++){

        uint rm_data = row_sum[to_server_id * gpu_n + *big_row] - server2server_data[to_server_id];
        for (auto small_row = smaller_row.begin(); small_row != smaller_row.end();){
            for (uint j = 0; j < gpu_n; j++){
                // check each element of the big row
                uint mv_data = MIN(MIN(rm_data, balanced_data[*big_row][to_server_id * gpu_n + j]), server2server_data[to_server_id] - row_sum[to_server_id * gpu_n + *small_row]);
                rm_data -= mv_data;
                // big row -> j ====> small row -> j via balance big row -> small row
                row_sum[to_server_id * gpu_n + *small_row] += mv_data;
                row_sum[to_server_id * gpu_n + *big_row] -= mv_data;
                balanced_data[*big_row][to_server_id * gpu_n + j] -= mv_data;
                balanced_data[*small_row][to_server_id * gpu_n + j] += mv_data;
                r.balance[ (*big_row) * gpu_n + (*small_row)] += mv_data;
            }
            if (rm_data == 0){
                break;
            }

            if (row_sum[to_server_id * gpu_n + *small_row] == server2server_data[to_server_id]){
                small_row = smaller_row.erase(small_row);
            }else{
                small_row ++;
            }
        }

        if (smaller_row.empty()){
            break;
        }
    }
}


void LocalScheduler::print_load_balance_step(struct load_balance_result *r){
    cout << "balance matrix: " << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            cout << setw(10);
            cout << r->balance[i * gpu_n + j];
        }
        cout << endl;
    }
    cout << "dispatch matrix: " << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            cout << setw(10);
            cout << r->dispatch[i * gpu_n + j];
        }
        cout << endl;
    }
}

void LocalScheduler::print(uint dst_server_id){
    cout << "server "<< server_id << " to server " << dst_server_id << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < gpu_n; j++){
            cout << setw(10);
            cout << data[i][dst_server_id * gpu_n + j];
        }
        cout << endl;
    }
}


void LocalScheduler::print(){
    uint dim = gpu_n * server_n;

    cout << "original matrix: " << endl;
    for (uint i = 0; i < gpu_n; i++){
        for (uint j = 0; j < dim; j++){
            cout << setw(10);
            cout << data[i][j];
        }
        cout << endl;
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
