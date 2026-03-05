#pragma once

#include <iostream>
#include "alltoall_matrix.h"
using namespace std;


struct hungarian_info_t{
    int matching[MAX_SERVER_NUM_DOUBLE];
    bool visit[MAX_SERVER_NUM_DOUBLE];
    uint row_to_col[MAX_SERVER_NUM][MAX_SERVER_NUM]; // the col vertices that each row vertice connects to
    uint row_to_col_n[MAX_SERVER_NUM];
};

void set_insert(uint val, uint * array, uint * sz);
void set_remove(uint val, uint * array, uint * sz);

struct map_data_t{
    uint key;
    uint val;
};

struct PermutationSet{
    uint64_t frequency;
    uint scaling_factor;
    uint dim;
    struct map_data_t mp[MAX_SERVER_NUM]; // mapping from row vertice to col vertice, both indexed from 0-dim-1
    uint mp_n;
};

void map_insert(struct map_data_t * array, uint * sz, uint key, uint val);
bool map_lookup(struct map_data_t * array, uint sz, uint key, uint * val);

void init_permutation_set(struct PermutationSet * ps, uint64_t _freq = 1, uint _sf = 1, uint _dim = 0);
void set_freq_permutation_set(struct PermutationSet * ps, uint64_t freq);
uint64_t get_freq_permutation_set(struct PermutationSet * ps);
void to_server_permutation_set(struct PermutationSet * ps, uint server_n, uint * r);
void from_server_permutation_set(struct PermutationSet * ps, uint server_n, uint * r);
void print_permutation_set(struct PermutationSet * ps);


struct FastAll2All{
    struct Matrix mat;     // original input matrix
    struct Matrix SDS_mat; // scaled doubly stochastic matrix
    struct hungarian_info_t hungarian_info;  //metadata used by hungarian algorithm
    struct PermutationSet p_sets[MAX_SERVER_NUM_SQUARE]; // permutation sets, storing decomposition results
    uint p_sets_ascending[MAX_SERVER_NUM_SQUARE];
    uint p_sets_n;
};

void init_fastall2all(struct FastAll2All * ata, Matrix * _mat);
void free_fastall2all(struct FastAll2All * ata);
void to_scaled_doubly_stochastic_matrix_fastall2all(struct FastAll2All * ata);
void update_edges_fastall2all(struct FastAll2All * ata);
uint update_permutation_sets_fastall2all(struct FastAll2All * ata);
void sort_permutation_sets(struct FastAll2All * ata);
void decompose_fastall2all(struct FastAll2All * ata);
uint hungarian_fastall2all(struct FastAll2All * ata);
bool hungarian_dfs_fastall2all(struct FastAll2All * ata, uint u);
bool verify_decomposition_fastall2all(struct FastAll2All * ata);
void print_decomposition_fastall2all(struct FastAll2All * ata);

