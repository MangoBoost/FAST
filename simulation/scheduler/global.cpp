#include "global.h"
#include "define.h"
#include <chrono>
#include <iostream>
using namespace chrono;
using namespace std;

GlobalScheduler::GlobalScheduler(uint _server_n, uint _gpu_n, vector<LocalScheduler*> _locals, INTER_LINK_TYPE type, INTRA_LINK_TYPE itype, bool scaling, double MBpu){
    server_n = _server_n;
    gpu_n = _gpu_n;
    locals = _locals;
    enable_scaling = scaling;
    inter_type = type;
    inter_info = get_inter_link_info(type);
    intra_type = itype;
    intra_info = get_intra_link_info(itype);
    if_approx = new bool[locals.size() * server_n];
    memset(if_approx, false, sizeof(bool) * locals.size() * server_n);

    // contruct the matrix from the local scheduler result
    uint *data = new uint[server_n * server_n];
    for (auto local = locals.begin(); local != locals.end(); local++){
       (*local) -> prepare_load_balance();
       uint src_svr = (*local) -> get_server_id();
       for (uint j = 0; j < server_n; j++){
            if_approx[src_svr * server_n + j] = (*local) -> check_balance_dispatch(j, MBpu);
            if (if_approx[src_svr * server_n + j]){
                (*local) -> server2server_approx(j);
            }
            data[src_svr * server_n + j] = (*local)->server2server_data[j];
       }
    }

    mat.copy(data, server_n);
    // cout << "Global scheduler prints server2server matrix: " << endl;
    // mat.print();
    delete[] data;
}

GlobalScheduler::GlobalScheduler(uint _server_n, uint _gpu_n, vector<LocalScheduler*> _locals, struct link_info_t _inter_link, struct link_info_t _intra_link, bool scaling, double MBpu){
    server_n = _server_n;
    gpu_n = _gpu_n;
    locals = _locals;
    enable_scaling = scaling;
    inter_info = _inter_link;
    intra_info = _intra_link;
    if_approx = new bool[locals.size() * server_n];
    memset(if_approx, false, sizeof(bool) * locals.size() * server_n);
    // contruct the matrix from the local scheduler result
    uint *data = new uint[server_n * server_n];
    for (auto local = locals.begin(); local != locals.end(); local++){
       (*local) -> prepare_load_balance();
       uint src_svr = (*local) -> get_server_id();
       for (uint j = 0; j < server_n; j++){
            if_approx[src_svr * server_n + j] = (*local) -> check_balance_dispatch(j, MBpu);
            if (if_approx[src_svr * server_n + j]){
                (*local) -> server2server_approx(j);
            }
            data[src_svr * server_n + j] = (*local)->server2server_data[j];
       }
    }
    mat.copy(data, server_n);
    // cout << "Global scheduler prints server2server matrix: " << endl;
    // mat.print();
    delete[] data;
}


GlobalScheduler::~GlobalScheduler(){
    delete[] if_approx;
}
struct pipeline_result_t GlobalScheduler::pipeline3(double MBpu, intra_transfer_topo_fn fn){
    FastAll2All all2all(&mat, gpu_n, inter_type, intra_type, MBpu);
    all2all.to_scaled_doubly_stochastic_matrix();
    all2all.decompose();

    // cout << "birkhoff decompostion succeed, pset_n:  " << all2all.p_sets.size() << " ,MBpu: " << MBpu <<endl;
    // //print first result
    // all2all.p_sets[0].print_permutation_matrix();
    // cout << "frequency: " << all2all.p_sets[0].get_freq() << endl;

    // OPTION 2: load balance in the begining (i.e., convert to balanced workload), only dispatch at each step
    uint locals_sz = locals.size(), pset_sz = all2all.p_sets.size();
    uint pid = 0;
    uint lid = 0;

    vector <vector<struct load_balance_result>> ds(pset_sz, vector<struct load_balance_result>(locals_sz, {.balance=NULL, .dispatch=NULL}));
    // dispatch for each pset
    for (pid = 0; pid != pset_sz; pid++){
        for (lid = 0; lid != locals_sz; lid ++){
            ds[pid][lid].dispatch = new uint[gpu_n * gpu_n];
            memset(ds[pid][lid].dispatch, 0,  sizeof(uint) * gpu_n * gpu_n);
        }
    }

    vector <vector<struct load_balance_result>> bs(locals_sz, vector<struct load_balance_result>(server_n, {.balance=NULL, .dispatch=NULL}));

    // balance only once
    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < server_n; s++){
            bs[lid][s].balance = new uint[gpu_n * gpu_n];
            memset(bs[lid][s].balance, 0, sizeof(uint) * gpu_n * gpu_n);
        }
    }
    
    // generate schedule for intra-server all2all - balance first
    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < locals_sz; s++){
           if (s == locals[lid]->get_server_id()){
            continue;
           }
           if (!if_approx[lid * server_n + s]){ // check the alpha and beta cost of balance and dispatch, do balance only if beta cost is dominant
                locals[lid]->server2server_balance(s, bs[lid][s]);
            }
            // else{
            //     cout << lid * server_n + s << endl;
            // }
        }
    }

    // generate schedule for intra-server all2all - dispatch for each step
    pid = 0;
    for (auto pset = all2all.p_sets.begin(); pset != all2all.p_sets.end(); pset++){
        lid = 0;
        for (auto local = locals.begin(); local != locals.end(); local ++){
            uint src_svr = (*local) -> get_server_id();
            auto lookup = (*pset).mp.find(src_svr);
            if (lookup == (*pset).mp.end()){
                LOG("error decomposition result");
                exit(1);
            }
            uint dst_svr = lookup -> second;
            // cout << "pid: " << pid << " , lid: "<< lid << " , src svr: " << src_svr << " , dst svr: " << dst_svr << endl;
            (*local) -> server2server_dispatch(dst_svr, ds[pid][lid], (*pset).get_freq());
            lid ++;
        }
        pid++;
    }
    // cout << "frequency: " << all2all.p_sets[0].get_freq() << endl;
    // uint dst_svr = all2all.p_sets[0].mp.find(0) -> second;
    // print_matrix(ds[0][0].dispatch, gpu_n, gpu_n);
    // locals[0]->print(dst_svr);

    // cout << "complete intra-tile all2all schedule" << endl;
    // initial intra-all2all and balance
    uint prev_pid = 0, cur_pid = 1, next_pid = 2;
    double intra1, intra2, intra_time, largest_intra_time, inter, balance_time;

    // evaluate the algorithm runtime
    double t = 0.0, bound = 0.0;

    // balance
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        balance_time = 0;
        for (uint s = 0; s < locals_sz; s++){
            if (s == locals[i]->get_server_id()){
                continue;
            }
            double cur_spd = (*fn)(bs[i][s].balance, gpu_n, locals[i]->get_link_info(), MBpu);
            balance_time += cur_spd;
            // cout << "first balance: "<< endl;
            // print_matrix(bs[i][s].balance, gpu_n, gpu_n);


        }
        largest_intra_time = MAX(largest_intra_time, balance_time);
    }
    t += largest_intra_time;

    // cout << "balance time: " << t << endl;
    // first inter transfer
    inter = permutation_set_cost(all2all.p_sets[0].get_freq(), MBpu);
    // intrinsic all2all
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra1 = (*fn)(locals[i]->get_intrinsic_all2all(), gpu_n, locals[i]->get_link_info(), MBpu);
        largest_intra_time = MAX(largest_intra_time, intra1);
    }

    t += MAX(inter, largest_intra_time);
    bound += inter;


    // pipeline the middle psets: inter transfer and dispatch
    for (cur_pid = 1, prev_pid = cur_pid - 1; 
    cur_pid < pset_sz;
    prev_pid = cur_pid, cur_pid++){

        largest_intra_time = 0;
        for (uint i = 0; i < locals_sz; i++){
            // previous step's intra-all2all dispatch
            if (cur_pid != 0){
                intra1 = (*fn)(ds[prev_pid][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
            }
            largest_intra_time = MAX(largest_intra_time, intra1);
        }

        // cout << cur_pid << " - " << t << " , intra: " << largest_intra_time<< " , inter: " << inter <<  " ,inter bigger: " << ((inter > largest_intra_time) ? "yes" : "no") << endl;
        // current step inter-all2all inter-server transfer
        inter = permutation_set_cost(all2all.p_sets[cur_pid].get_freq(), MBpu);
        t += MAX(largest_intra_time, inter);
        bound += inter;
    }

    // dispatch for the last pset
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra1 = (*fn)(ds[pset_sz-1][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
        largest_intra_time = MAX(largest_intra_time, intra1);
    }
    t += largest_intra_time;

    // cout << "FastAll2All time: "<< t << " ,bound: "<< bound << " ,extra overhead: " << (t - bound) / bound * 100 << " %"<<endl;

    // cout << "release intra-all2all memory allocation" << endl;
    // free memory for intra all2all
    for(pid = 0; pid < pset_sz; pid ++){
        for (lid = 0; lid < locals_sz; lid++){
            delete[] ds[pid][lid].dispatch;
        }
    }

    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < locals_sz; s++){
            delete[] bs[lid][s].balance;
        }
    }

    // cout << "complete release intra-all2all memory allocation" << endl;
    double ratio = (t - bound) / t;
    struct pipeline_result_t pipe_r = {.t = t, .ratio = ratio};
    return pipe_r;
}


struct pipeline_result_t GlobalScheduler::pipeline2(double MBpu, scaling_success_condition fn){
    FastAll2All all2all(&mat, gpu_n, inter_type, intra_type, MBpu);
    if (enable_scaling){
        all2all.to_scaled_matrix(fn);
    }
    all2all.to_scaled_doubly_stochastic_matrix();
    all2all.decompose();

    // cout << "birkhoff decompostion succeed, pset_n:  " << all2all.p_sets.size() << " ,MBpu: " << MBpu <<endl;
    // //print first result
    // all2all.p_sets[0].print_permutation_matrix();
    // cout << "frequency: " << all2all.p_sets[0].get_freq() << endl;

    // OPTION 2: load balance in the begining (i.e., convert to balanced workload), only dispatch at each step
    uint locals_sz = locals.size(), pset_sz = all2all.p_sets.size();
    uint pid = 0;
    uint lid = 0;

    vector <vector<struct load_balance_result>> ds(pset_sz, vector<struct load_balance_result>(locals_sz, {.balance=NULL, .dispatch=NULL}));
    // dispatch for each pset
    for (pid = 0; pid != pset_sz; pid++){
        for (lid = 0; lid != locals_sz; lid ++){
            ds[pid][lid].dispatch = new uint[gpu_n * gpu_n];
            memset(ds[pid][lid].dispatch, 0,  sizeof(uint) * gpu_n * gpu_n);
        }
    }

    vector <vector<struct load_balance_result>> bs(locals_sz, vector<struct load_balance_result>(server_n, {.balance=NULL, .dispatch=NULL}));

    // balance only once
    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < server_n; s++){
            bs[lid][s].balance = new uint[gpu_n * gpu_n];
            memset(bs[lid][s].balance, 0, sizeof(uint) * gpu_n * gpu_n);
        }
    }
    
    // generate schedule for intra-server all2all - balance first
    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < locals_sz; s++){
           if (s == locals[lid]->get_server_id()){
            continue;
           }
           if (!if_approx[lid * server_n + s]){ // check the alpha and beta cost of balance and dispatch, do balance only if beta cost is dominant
                locals[lid]->server2server_balance(s, bs[lid][s]);
            }
            // else{
            //     cout << lid * server_n + s << endl;
            // }
        }
    }

    // generate schedule for intra-server all2all - dispatch for each step
    pid = 0;
    for (auto pset = all2all.p_sets.begin(); pset != all2all.p_sets.end(); pset++){
        lid = 0;
        for (auto local = locals.begin(); local != locals.end(); local ++){
            uint src_svr = (*local) -> get_server_id();
            auto lookup = (*pset).mp.find(src_svr);
            if (lookup == (*pset).mp.end()){
                LOG("error decomposition result");
                exit(1);
            }
            uint dst_svr = lookup -> second;
            // cout << "pid: " << pid << " , lid: "<< lid << " , src svr: " << src_svr << " , dst svr: " << dst_svr << endl;
            (*local) -> server2server_dispatch(dst_svr, ds[pid][lid], (*pset).get_freq());
            lid ++;
        }
        pid++;
    }
    // cout << "frequency: " << all2all.p_sets[0].get_freq() << endl;
    // uint dst_svr = all2all.p_sets[0].mp.find(0) -> second;
    // print_matrix(ds[0][0].dispatch, gpu_n, gpu_n);
    // locals[0]->print(dst_svr);

    // cout << "complete intra-tile all2all schedule" << endl;
    // initial intra-all2all and balance
    uint prev_pid = 0, cur_pid = 1, next_pid = 2;
    double intra1, intra2, intra_time, largest_intra_time, inter, balance_time;

    // evaluate the algorithm runtime
    double t = 0.0, bound = 0.0;

    // balance
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        balance_time = 0;
        for (uint s = 0; s < locals_sz; s++){
            if (s == locals[i]->get_server_id()){
                continue;
            }
            double cur_spd = spread_out(bs[i][s].balance, gpu_n, locals[i]->get_link_info(), MBpu);
            balance_time += cur_spd;
            // cout << "first balance: "<< endl;
            // print_matrix(bs[i][s].balance, gpu_n, gpu_n);


        }
        largest_intra_time = MAX(largest_intra_time, balance_time);
    }
    t += largest_intra_time;

    // cout << "balance time: " << t << endl;
    // first inter transfer
    inter = permutation_set_cost(all2all.p_sets[0].get_freq(), MBpu);
    // intrinsic all2all
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra1 = spread_out(locals[i]->get_intrinsic_all2all(), gpu_n, locals[i]->get_link_info(), MBpu);
        largest_intra_time = MAX(largest_intra_time, intra1);
    }

    t += MAX(inter, largest_intra_time);
    bound += inter;


    // pipeline the middle psets: inter transfer and dispatch
    for (cur_pid = 1, prev_pid = cur_pid - 1; 
    cur_pid < pset_sz;
    prev_pid = cur_pid, cur_pid++){

        largest_intra_time = 0;
        for (uint i = 0; i < locals_sz; i++){
            // previous step's intra-all2all dispatch
            if (cur_pid != 0){
                intra1 = spread_out(ds[prev_pid][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
            }
            largest_intra_time = MAX(largest_intra_time, intra1);
        }

        // cout << cur_pid << " - " << t << " , intra: " << largest_intra_time<< " , inter: " << inter <<  " ,inter bigger: " << ((inter > largest_intra_time) ? "yes" : "no") << endl;
        // current step inter-all2all inter-server transfer
        inter = permutation_set_cost(all2all.p_sets[cur_pid].get_freq(), MBpu);
        t += MAX(largest_intra_time, inter);
        bound += inter;
    }

    // dispatch for the last pset
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra1 = spread_out(ds[pset_sz-1][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
        largest_intra_time = MAX(largest_intra_time, intra1);
    }
    t += largest_intra_time;

    // cout << "FastAll2All time: "<< t << " ,bound: "<< bound << " ,extra overhead: " << (t - bound) / bound * 100 << " %"<<endl;

    // cout << "release intra-all2all memory allocation" << endl;
    // free memory for intra all2all
    for(pid = 0; pid < pset_sz; pid ++){
        for (lid = 0; lid < locals_sz; lid++){
            delete[] ds[pid][lid].dispatch;
        }
    }

    for (lid = 0; lid < locals_sz; lid++){
        for (uint s = 0; s < locals_sz; s++){
            delete[] bs[lid][s].balance;
        }
    }

    // cout << "complete release intra-all2all memory allocation" << endl;
    double ratio = (t - bound) / t;
    struct pipeline_result_t pipe_r = {.t = t, .ratio = ratio};
    return pipe_r;
}

struct pipeline_result_t GlobalScheduler::pipeline(double MBpu){
    // return the algorithm time cost for the traffic demand matrix
    // server-level scheudling
    // cout << "start birkhoff decompostion" << endl;
    FastAll2All all2all(&mat, gpu_n, inter_type, intra_type, MBpu);
    if (enable_scaling){
        all2all.to_scaled_matrix(&balance_alpha_beta);
    }
    all2all.to_scaled_doubly_stochastic_matrix();
    all2all.decompose();

    // cout << "birkhoff decompostion succeed, pset_n:  " << all2all.p_sets.size()<<endl;

    // OPTION 1: load balance at each step
    uint locals_sz = locals.size(), pset_sz = all2all.p_sets.size();
    uint pid = 0;
    uint lid = 0;
    vector <vector<struct load_balance_result>> lbs(pset_sz, vector<struct load_balance_result>(locals_sz, {.balance=NULL, .dispatch=NULL}));

    for (pid = 0; pid != pset_sz; pid++){
        for (lid = 0; lid != locals_sz; lid ++){
            lbs[pid][lid].balance = new uint[gpu_n * gpu_n];
            lbs[pid][lid].dispatch = new uint[gpu_n * gpu_n];
            memset(lbs[pid][lid].balance, 0, sizeof(uint) * gpu_n * gpu_n);
            memset(lbs[pid][lid].dispatch, 0,  sizeof(uint) * gpu_n * gpu_n);
        }
    }
    
    // cout << "complete intra-all2all memory allocation" << endl;

    // generate schedule for intra-server all2all
    //TODO: sort the permutation sets based on the frequency
    pid = 0;
    for (auto pset = all2all.p_sets.begin(); pset != all2all.p_sets.end(); pset++){
        lid = 0;
        for (auto local = locals.begin(); local != locals.end(); local ++){
            uint src_svr = (*local) -> get_server_id();
            auto lookup = (*pset).mp.find(src_svr);
            if (lookup == (*pset).mp.end()){
                LOG("error decomposition result");
                exit(1);
            }
            uint dst_svr = lookup -> second;
            // cout << "pid: " << pid << " , lid: "<< lid << " , src svr: " << src_svr << " , dst svr: " << dst_svr << endl;
            (*local) -> server2server_balance(dst_svr, lbs[pid][lid], (*pset).get_freq());
            lid ++;
        }
        pid++;
    }

    // cout << "complete intra-tile all2all schedule" << endl;
    // initial intra-all2all and balance
    uint prev_pid = 0, cur_pid = 1, next_pid = 2;
    double intra1, intra2, intra_time, largest_intra_time, inter, largest_intra1, largest_intra2;

    // evaluate the algorithm runtime
    double t = 0.0, bound = 0.0;

    // first pset
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra2 = spread_out(lbs[0][i].balance, gpu_n, locals[i]->get_link_info(), MBpu);
        // cout << "local all2all " << i << " : " << intra2 <<endl;
        // print_matrix(lbs[0][i].balance, gpu_n, gpu_n);
        largest_intra_time = MAX(largest_intra_time, intra2);
    }
    t += largest_intra_time;

    // pipeline the middle psets 
    for (cur_pid = 0, next_pid = cur_pid + 1, prev_pid = cur_pid - 1; 
    cur_pid < pset_sz;
    prev_pid = cur_pid, cur_pid = next_pid, next_pid++){

        largest_intra_time = 0;
        largest_intra1 = 0;
        largest_intra2 = 0;
        for (uint i = 0; i < locals_sz; i++){
            // previous step's intra-all2all dispatch
            if (cur_pid != 0){
                intra1 = spread_out(lbs[prev_pid][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
            }
            // next step's intra-all2all balance
            if (cur_pid != pset_sz - 1){
                intra2 = spread_out(lbs[next_pid][i].balance, gpu_n, locals[i]->get_link_info(), MBpu);
            }else{
                // the initial intrinsic intra all2all
                intra2 = spread_out(locals[i]->get_intrinsic_all2all(), gpu_n, locals[i]->get_link_info(), MBpu);
            }
            intra_time = intra1 + intra2;
            largest_intra_time = MAX(largest_intra_time, intra_time);
            largest_intra1 = MAX(largest_intra1, intra1);
            largest_intra2 = MAX(largest_intra2, intra2);
        }

        // cout << cur_pid << " - " << t << " , intra total: " << largest_intra_time << " , intra1: " << largest_intra1<< " , intra2: " << largest_intra2 << " , inter: " << inter <<  " ,inter bigger: " << ((inter > largest_intra_time) ? "yes" : "no") << endl;
        // current step inter-all2all inter-server transfer
        inter = permutation_set_cost(all2all.p_sets[cur_pid].get_freq(), MBpu);
        t += MAX(largest_intra_time, inter);
        bound += inter;
    }

    // dispatch for the last pset
    largest_intra_time = 0;
    for (uint i = 0; i < locals_sz; i++){
        intra1 = spread_out(lbs[pset_sz-1][i].dispatch, gpu_n, locals[i]->get_link_info(), MBpu);
        largest_intra_time = MAX(largest_intra_time, intra1);
    }

    t += largest_intra_time;

    // cout << "FastAll2All time: "<< t << " ,bound: "<< bound << " ,extra overhead: " << (t - bound) / bound * 100 << " %"<<endl;

    // cout << "release intra-all2all memory allocation" << endl;
    // free memory for intra all2all
    for(pid = 0; pid < pset_sz; pid ++){
        for (lid = 0; lid < locals_sz; lid++){
            delete[] lbs[pid][lid].balance;
            delete[] lbs[pid][lid].dispatch;
        }
    }

    // cout << "complete release intra-all2all memory allocation" << endl;
    double ratio = (t - bound) / t;
    struct pipeline_result_t pipe_r = {.t = t, .ratio = ratio};
    return pipe_r;
}

double GlobalScheduler::permutation_set_cost(uint freq, double MBpu){
    return inter_info.alpha + freq * inter_info.beta * MBpu;
}
