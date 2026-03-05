#pragma once

#include <iostream>
#include <vector>
#include <unordered_set>
#include <map>
#include "matrix.h"
#include "config.h"
using namespace std;


struct hungarian_info_t{
    vector<int> matching;
    vector<bool> visit;
    vector<unordered_set<uint> > row_to_col; // the col vertices that each row vertice connects to
};


class PermutationSet{
private:
    uint frequency;
    uint scaling_factor;
    uint dim;
public:
    map<uint, uint> mp;     // mapping between row vertice and col vertice, both indexed from 0-dim-1
    PermutationSet(uint _freq = 1, uint _sf = 1, uint _dim = 0):frequency(_freq), scaling_factor(_sf), dim(_dim){}
    ~PermutationSet(){}
    void set_freq(uint freq){frequency = freq;}
    uint get_freq(){return frequency * scaling_factor;}
    void print_permutation_matrix();
};

struct scale_info_t{
    uint max_sum;
    uint factor;
};

typedef bool (*scaling_success_condition)(uint, uint, struct scale_info_t, struct link_info_t, struct link_info_t, double);
bool limit_max_sum(struct scale_info_t s_info);

class FastAll2All{
private:
    Matrix mat;     // original input matrix
    Matrix S_mat;   // scaled matrix
    Matrix SDS_mat; // scaled doubly stochastic matrix
    struct hungarian_info_t hungarian_info;  //metadata used by hungarian algorithm
    bool is_scaled;
    uint scaling_factor;
    double MB_per_unit;
    INTER_LINK_TYPE inter_type;
    struct link_info_t inter_link;
    INTRA_LINK_TYPE intra_type;
    struct link_info_t intra_link;
    uint gpu_n;

public:
    vector<PermutationSet> p_sets; // permutation sets, storing decomposition results
    FastAll2All(Matrix * _mat, uint _gpu_n, INTER_LINK_TYPE _type = INFB, INTRA_LINK_TYPE _itype = DGX2, double MBpu = 1.0);
    ~FastAll2All(){}
    void print(){is_scaled ? S_mat.print() : mat.print();SDS_mat.print();}
    struct scale_info_t scale_matrix(uint factor, uint* buffer);
    int scaling_binary_search(scaling_success_condition fn, int l, int r, uint step, uint* buffer);
    void to_scaled_matrix(scaling_success_condition fn);
    void to_scaled_doubly_stochastic_matrix();
    void update_edges();
    uint update_permutation_sets();
    void decompose();
    uint hungarian();
    bool hungarian_dfs(uint u);
    bool verify_decomposition();
    void print_decomposition();
};

bool limit_max_sum(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu);
bool balance_alpha_beta(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu);
bool always_scale(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu);