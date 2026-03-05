#pragma once
#include <iostream>
#include <vector>
using namespace std;


enum INTRA_LINK_TYPE{
    DGX2, MI300X, NDV2, FAST, H100, B100
};

enum INTER_LINK_TYPE{
    ETHER100, ETHER400, INFB
};

struct link_info_t{
    double tput;    //
    double alpha;   // us
    double beta;    // us/MB
};


struct link_info_t get_intra_link_info(INTRA_LINK_TYPE tyoe);
struct link_info_t get_inter_link_info(INTER_LINK_TYPE type);

double spread_out_baseline(uint * data, uint server_n, uint gpu_n, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu = 1.0);


typedef double (*intra_transfer_topo_fn)(uint *, uint, struct link_info_t, double);
double spread_out(uint * data, uint dim, struct link_info_t link, double MBpu = 1.0);
double intra_transfer_full_mesh(uint * data,  uint dim, struct link_info_t link, double MBpu = 1.0);
double intra_transfer_hybrid_cude_mesh(uint * data,  uint dim, struct link_info_t link, double MBpu = 1.0);
double intra_transfer_2ring(uint * data,  uint dim, struct link_info_t link, double MBpu = 1.0);



void print_matrix(uint * data, uint m, uint n);

struct statistics_t{
    double avg;
    double err;
};

struct simulation_result_t{
    double flash_algbw;
    double opt_algbw;
    double spread_algbw;
};

void compute_average_standard_deviation(vector<double> *vec, double * avg, double * sd);
double Gbps_to_us_per_MB(uint speed);