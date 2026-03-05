#pragma once

#include <iostream>
#include "config.h"
#include "define.h"
#include <map>

using namespace std;


struct load_balance_result{
    uint * balance;
    uint * dispatch; 
};

class LocalScheduler{
private:
    uint ** data;
    uint ** balanced_data;
    uint * intrinsic_all2all;
    uint gpu_n;
    uint server_n;
    uint server_id;
    enum INTRA_LINK_TYPE intra_type;
    struct link_info_t intra_info;
    map<uint ,struct load_balance_result> intra_all2all; //destination server id ----> intra_all2all_ops
    uint * row_sum;

public:
    uint * server2server_data;
    LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id, INTRA_LINK_TYPE _type = DGX2);
    LocalScheduler(uint* _data, uint _gpu_n, uint _server_n, uint _server_id, struct link_info_t _intra_link);
    ~LocalScheduler();
    void load_balance();
    bool check_balance_dispatch(uint to_server_id, double MBpu = 1.0);
    void server2server_approx(uint to_server_id);
    void server2server_balance(uint to_server_id, struct load_balance_result r);
    void prepare_load_balance();
    void server2server_balance(uint to_server_id, struct load_balance_result r, uint freq);
    void server2server_dispatch(uint to_server_id, struct load_balance_result r, uint freq);
    uint get_server_id(){return server_id;}
    struct link_info_t get_link_info(){return intra_info;}
    uint * get_intrinsic_all2all(){return intrinsic_all2all;}
    void print();
    void print(uint dst_server_id);
    void print_load_balance_step(struct load_balance_result *r);
};