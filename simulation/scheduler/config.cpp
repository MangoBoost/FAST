#include "config.h"
#include "define.h"
#include <iomanip>
#include <cmath>

using namespace std;

struct link_info_t get_intra_link_info(INTRA_LINK_TYPE type){
    struct link_info_t r;
    switch (type){
        case DGX2:
            r.alpha = 0.1;
            r.beta = 8; //TODO: check specs and measure
            break;
        case FAST:
            r.alpha = 0.1;
            r.beta = 1; //TODO: check specs and measure
            break;
        case B100:
            r.alpha = 0.1;
            r.beta = 1.2; //TODO: check specs and measure
            break;
        case H100:
            r.alpha = 0.1;
            r.beta = 2.3; //TODO: check specs and measure
            break;
        case NDV2:
            r.alpha = 0.1;
            r.beta = 46;
            break;
        case MI300X:
            //TODO: check MI300X specs
            r.alpha = 0.1;
            r.beta = 5;
            break;
    }

    return r;
}

struct link_info_t get_inter_link_info(INTER_LINK_TYPE type){
    struct link_info_t r;
    switch (type){
        case INFB:
            r.alpha = 1.7;
            r.beta = 106;
            break;
        case ETHER100:
            //TODO: measure ethernet specs
            r.alpha = 0.7;
            r.beta = 84;
            break;
        case ETHER400:
            r.alpha = 0.7;
            r.beta = 22;
            break;
    }
    return r;
}

double spread_out(uint * data, uint dim, struct link_info_t link, double MBpu){
    double r = 0.0;
    uint largest_in_diagnal = 0;
    for (uint i = 0; i < dim; i ++){
        largest_in_diagnal = 0;
        for (uint j = 0; j < dim; j++){
            largest_in_diagnal = MAX(largest_in_diagnal, data[j * dim + ((j + i) % dim)]);
        }
        if (largest_in_diagnal != 0){
            r += link.alpha +  largest_in_diagnal * link.beta * MBpu;
        }
    }
    return r;
}

double intra_transfer_full_mesh(uint * data, uint dim, struct link_info_t link, double MBpu){
    uint max_ele = 0;
    for (uint i = 0; i < dim; i ++){
        for (uint j = 0; j < dim; j++){
            max_ele = MAX(max_ele, data[i * dim + j]);
        }
    }
    return link.alpha +  max_ele * link.beta * MBpu;
}

double intra_transfer_hybrid_cude_mesh(uint * data, uint dim, struct link_info_t link, double MBpu){
//50 GBps bidirectional -> 25 GBps duplex
//      0    1     4    5 
//      2    3     6    7
    // first do transfer at small ring
    double result = 0;
    double r[32];
    r[0] = link.alpha +   data[0 * dim + 1] * link.beta / 2 * MBpu;
    r[1] = link.alpha +   data[0 * dim + 2] * link.beta / 2 * MBpu;
    r[2] = link.alpha +   data[0 * dim + 3] * link.beta * MBpu;
    r[3] = link.alpha +   data[1 * dim + 0] * link.beta / 2 * MBpu;
    r[4] = link.alpha +   data[1 * dim + 2] * link.beta * MBpu;
    r[5] = link.alpha +   data[1 * dim + 3] * link.beta * MBpu;
    r[6] = link.alpha +   data[2 * dim + 0] * link.beta / 2 * MBpu;
    r[7] = link.alpha +   data[2 * dim + 1] * link.beta * MBpu;
    r[8] = link.alpha +   data[2 * dim + 3] * link.beta / 2 * MBpu;
    r[9] = link.alpha +   data[3 * dim + 0] * link.beta * MBpu;
    r[10] = link.alpha +  data[3 * dim + 1] * link.beta * MBpu;
    r[11] = link.alpha +   data[3 * dim + 2] * link.beta / 2 * MBpu;
    r[12] = link.alpha +   data[0 * dim + 5] * link.beta * MBpu;
    r[13] = link.alpha +   data[1 * dim + 4] * link.beta / 2 * MBpu;
    r[14] = link.alpha +   data[2 * dim + 7] * link.beta * MBpu;
    r[15] = link.alpha +   data[3 * dim + 6] * link.beta / 2 * MBpu;
    r[16] = link.alpha +   data[5 * dim + 0] * link.beta * MBpu;
    r[17] = link.alpha +   data[4 * dim + 1] * link.beta / 2 * MBpu;
    r[18] = link.alpha +   data[7 * dim + 2] * link.beta * MBpu;
    r[19] = link.alpha +   data[6 * dim + 3] * link.beta / 2 * MBpu;
    r[20] = link.alpha +   data[4 * dim + 5] * link.beta / 2 * MBpu;
    r[21] = link.alpha +   data[4 * dim + 6] * link.beta * MBpu;
    r[22] = link.alpha +   data[4 * dim + 7] * link.beta * MBpu;
    r[23] = link.alpha +   data[5 * dim + 4] * link.beta / 2 * MBpu;
    r[24] = link.alpha +   data[5 * dim + 6] * link.beta * MBpu;
    r[25] = link.alpha +   data[5 * dim + 7] * link.beta / 2 * MBpu;
    r[26] = link.alpha +   data[6 * dim + 4] * link.beta * MBpu;
    r[27] = link.alpha +   data[6 * dim + 5] * link.beta * MBpu;
    r[28] = link.alpha +   data[6 * dim + 7] * link.beta / 2 * MBpu;
    r[29] = link.alpha +   data[7 * dim + 4] * link.beta * MBpu;
    r[30] = link.alpha +  data[7 * dim + 5] * link.beta / 2 * MBpu;
    r[31] = link.alpha +   data[7 * dim + 6] * link.beta / 2 * MBpu;

    double first_step_max = 0;
    for (uint i = 0; i < 32; i ++){
        first_step_max = MAX(first_step_max, r[i]);
    }
    result += first_step_max;
    // transfer via intermediate hop 
    double t[8];
    //      0    1     4    5 
    //      2    3     6    7
    t[0] = 2 * link.alpha +  MAX(MAX(data[0 * dim + 4] / 2, data[2 * dim + 4]), data[3 * dim + 4]) * link.beta * MBpu + (data[0 * dim + 4] + data[2 * dim + 4] + data[3 * dim + 4]) * link.beta / 2 * MBpu;
    t[1] = 2 * link.alpha +  MAX(MAX(data[1 * dim + 5] / 2, data[2 * dim + 5] / 2), data[3 * dim + 5]) * link.beta * MBpu + (data[1 * dim + 5] + data[2 * dim + 5] + data[3 * dim + 5]) * link.beta * MBpu;
    t[2] = 2 * link.alpha +  MAX(MAX(data[0 * dim + 6], data[1 * dim + 6]), data[2 * dim + 6] / 2) * link.beta * MBpu + (data[0 * dim + 6] + data[1 * dim + 6] + data[2 * dim + 6]) * link.beta / 2 * MBpu;
    t[3] = 2 * link.alpha +  MAX(MAX(data[0 * dim + 7] / 2, data[1 * dim + 7]), data[3 * dim + 7] / 2) * link.beta * MBpu + (data[0 * dim + 7] + data[1 * dim + 7] + data[3 * dim + 7]) * link.beta * MBpu;
    t[4] = 2 * link.alpha +  MAX(MAX(data[4 * dim + 0] / 2, data[6 * dim + 0]), data[7 * dim + 0]/2) * link.beta * MBpu + (data[4 * dim + 0] + data[6 * dim + 0] + data[7 * dim + 0]) * link.beta * MBpu;
    t[5] = 2 * link.alpha +  MAX(MAX(data[5 * dim + 1] / 2, data[6 * dim + 1]), data[7 * dim + 1]) * link.beta * MBpu + (data[5 * dim + 1] + data[6 * dim + 1] + data[7 * dim + 1]) * link.beta / 2 * MBpu;
    t[6] = 2 * link.alpha +  MAX(MAX(data[4 * dim + 2], data[5 * dim + 2] / 2), data[6 * dim + 2] /2) * link.beta * MBpu + (data[4 * dim + 2] + data[5 * dim + 2] + data[6 * dim + 2]) * link.beta * MBpu;
    t[7] = 2 * link.alpha +  MAX(MAX(data[4 * dim + 3], data[5 * dim + 3]), data[7 * dim + 3] /2) * link.beta * MBpu + (data[4 * dim + 3] + data[5 * dim + 3] + data[7 * dim + 3]) * link.beta / 2 * MBpu;
    double second_step_max = 0;
    for (uint i = 0; i < 8; i ++ ){
        second_step_max = MAX(second_step_max, t[i]);
    }
    result += second_step_max;
    return result;
}


double intra_transfer_2ring(uint * data, uint dim, struct link_info_t link, double MBpu){
//100 GBps bidirectional -> 50 GBps duplex
// 2 rings
double first_ring = 0;
for (uint i = 1; i < 4; i++){
    uint largest_in_diagnal = 0;
    for (uint j = 0; j < dim; j++){
        largest_in_diagnal = MAX(largest_in_diagnal, data[j * dim + ((j + i) % dim)]);
    }

    if (largest_in_diagnal != 0){
        first_ring += (link.alpha +  largest_in_diagnal * link.beta * MBpu) * i;
    }
}
double reverse_ring = 0;
for (uint i = 1; i < 5; i++){
    uint largest_in_diagnal = 0;
    for (uint j = 0; j < dim; j++){
        largest_in_diagnal = MAX(largest_in_diagnal, data[j * dim + ((j + dim - i) % dim)]);
    }

    if (largest_in_diagnal != 0){
        reverse_ring += (link.alpha +  largest_in_diagnal * link.beta * MBpu) * i;
    }
}

return MAX(first_ring, reverse_ring);
}


double spread_out_baseline(uint * data, uint server_n, uint gpu_n, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu){
    double r = 0.0;
    uint dim = server_n * gpu_n;
    double current_sending = 0.0, largest_sending = 0.0;
    double alpha = 0, beta = 0;
    for (uint i = 0; i < dim; i++){
        largest_sending = 0;
        for (uint j = 0; j < dim; j++){
            uint row_id = j, col_id = (j + i) % dim;
            if (data[row_id * dim + col_id] != 0){
                if (row_id / gpu_n == col_id / gpu_n){
                    // intra server links
                    current_sending = intra_link.alpha +  data[row_id * dim + col_id] * intra_link.beta * MBpu;
                }else{
                    // inter server links
                    current_sending = inter_link.alpha +  data[row_id * dim + col_id] * inter_link.beta * MBpu;
                }
                largest_sending = MAX(largest_sending, current_sending);
            }
        }
        alpha += inter_link.alpha;
        beta += largest_sending - inter_link.alpha;
        r += largest_sending;
    }
    // cout << "alpha: " << alpha << ", beta: " << beta <<endl;
    return r;
}


void print_matrix(uint * data, uint m, uint n){ //width m, height n
    cout << "--------------------------------------"<<endl;
    for(uint i = 0; i < n; i ++){
        for (uint j = 0; j < m; j++){
            cout << setw(10);
            cout << data[i * m + j];
        }
        cout << endl;
    }
    cout << "--------------------------------------"<<endl;
}

void compute_average_standard_deviation(vector<double> *vec, double * avg, double * sd){
    double sum = 0.0, mean = 0.0, standard_deviation = 0.0;
    uint sz = vec->size();
    for(uint i = 0; i < sz; i++){
        sum += (*vec)[i];
    }
    mean = sum / (double)sz;
    for (uint i = 0; i < sz; i++){
        standard_deviation += pow((*vec)[i] - mean, 2);
    }
    (*avg) = mean;
    (*sd) = sqrt(standard_deviation / sz);
}

double Gbps_to_us_per_MB(uint speed){
    return (double) (8 * 1e3) / (double) speed;
}
