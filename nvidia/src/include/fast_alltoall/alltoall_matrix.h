#pragma once

#include <iostream>
#include "alltoall_define.h"

struct row_col_info_t {
    uint idx;
    uint64_t sum;
};

struct sdsm_info_t{
    bool is_sdsm;
    uint64_t max_row_col_sum;
    struct row_col_info_t non_max_row[MAX_SERVER_NUM];
    uint non_max_row_n;
    struct row_col_info_t non_max_col[MAX_SERVER_NUM];
    uint non_max_col_n;
};


struct Matrix {
    uint64_t ** data;
    uint dim;
    uint unit;
    struct sdsm_info_t sdsm_info;
};

void init_matrix(struct Matrix *m, uint _dim = 0);
void update_matrix(struct Matrix *m);
void free_matrix(struct Matrix *m);
void copy_matrix(struct Matrix *m, uint64_t * _data, uint source_dim);
void copy_matrix(struct Matrix *dst, struct Matrix * src);
bool valid_matrix(struct Matrix *m);
bool valid_sdsm_matrix(struct Matrix *m);
uint64_t get_matrix(struct Matrix *m, uint x, uint y);
bool set_matrix(struct Matrix *m, uint64_t val, uint x, uint y);
bool add_matrix(struct Matrix *m, uint64_t val, uint x, uint y);
bool subtract_matrix(struct Matrix *m, uint64_t val, uint x, uint y);
void scale_matrix(struct Matrix *m, uint factor);
bool equal_to_matrix(struct Matrix *a, struct Matrix *b);
void get_sdsm_info_matrix(struct Matrix *m);

