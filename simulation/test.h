#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib> 
#include "config.h"
#include "all2all.h"

using namespace std;


#define MAX_ELEMENT 10000


//-----------------------------------------------------------------------------------------------------------------------
// Zipf distribution
// Modified from: https://github.com/nal-epfl/castan/blob/master/scripts/pcap_tools/create_zipfian_distribution_pcap.py
//-----------------------------------------------------------------------------------------------------------------------
class zipf_distribution{
private:
    double s;   // skewness parameter 0 < s < 1
public:
    zipf_distribution(double _s): s(_s){srand(time(NULL));}
    ~zipf_distribution(){}
    uint32_t zipf_inverse_cdf_fast(double p, uint32_t N);
    void zipf(vector<uint> * r, uint N);
};


class DecompositionTester{
private:
    uint times_per_dim;
    vector<uint> test_dims;

public:
    DecompositionTester(vector<uint> _test_dims, uint _times_per_dim = 10): test_dims(_test_dims), times_per_dim(_times_per_dim){ srand((unsigned)time(0));}
    ~DecompositionTester(){}
    void run(bool enable_scaling=false);
};

class FastAll2AllTester{
private:
    uint test_times;
    bool enable_scaling;
    uint server_n;
    uint gpu_n;
    INTER_LINK_TYPE inter_link_type;
    INTRA_LINK_TYPE intra_link_type;

public:
    FastAll2AllTester(uint s_n, uint g_n, uint times = 10, bool scaling=false, INTER_LINK_TYPE inter_type=INFB, INTRA_LINK_TYPE intra_type=DGX2);
    ~FastAll2AllTester(){}
    void run();
    void server_gpu_number_benchmark(INTRA_LINK_TYPE _intra_type = DGX2, INTER_LINK_TYPE _inter_type = INFB);
    void fabric_speed_benchmark();
    void topology_benchmark(intra_transfer_topo_fn fn, uint inter_Gbps, uint intra_Gbps);
    void skewness_benchmark(INTRA_LINK_TYPE _intra_type=DGX2, INTER_LINK_TYPE _inter_type=INFB);
    void transfer_size_benchmark(bool enable_scaling, INTRA_LINK_TYPE _intra_type=DGX2, INTER_LINK_TYPE _inter_type=INFB, scaling_success_condition fn = &balance_alpha_beta, string fstr = "0");
};

struct server_gpu_config_t{
    uint svr_n;
    uint gpu_n;
    bool operator<(const server_gpu_config_t &o) const {
    return svr_n < o.svr_n || (svr_n == o.svr_n && gpu_n < o.gpu_n);
}
};

struct server_speed_config_t{
    uint svr_n;
    uint speed;
    double ratio;
    bool operator<(const server_speed_config_t &o) const {
    return svr_n < o.svr_n || (svr_n == o.svr_n && speed < o.speed);
    }
};

struct server_skewness_config_t{
    uint svr_n;
    double s;
    bool operator<(const server_skewness_config_t &o) const {
    return svr_n < o.svr_n || (svr_n == o.svr_n && s < o.s);
}
};

struct server_transfer_config_t{
    uint svr_n;
    double sz;
    bool operator<(const server_transfer_config_t &o) const {
    return svr_n < o.svr_n || (svr_n == o.svr_n && sz < o.sz);
}
};