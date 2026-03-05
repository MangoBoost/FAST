#include <iostream>
#include <stdio.h>
#include <iomanip>
#include "fast_alltoall/alltoall_matrix.h"
#include <hip/hip_runtime.h>


void init_matrix(struct Matrix *m, uint _dim){
    m->data = NULL;
    m->dim = _dim;
    m->unit = 1;
    m->sdsm_info.is_sdsm = false;
    m->sdsm_info.max_row_col_sum = 0;
    if (_dim > 0){
        m->data = (uint64_t **) malloc(sizeof(uint64_t*) * _dim);
        // hipMallocManaged((void**) &m->data, sizeof(uint*) * _dim);
        for (uint i = 0; i < _dim; i++){
            m->data[i] = (uint64_t *) malloc(sizeof(uint64_t) * _dim);
            // hipMallocManaged((void**) &m->data[i], sizeof(uint) * _dim);
            for (uint j = 0; j < _dim; j++){
                m->data[i][j] = 0;
            }
        }
    }
}

void update_matrix(struct Matrix *m){
    m->unit = 1;
    m->sdsm_info.is_sdsm = false;
    m->sdsm_info.max_row_col_sum = 0;
    for (uint i = 0; i < m->dim; i ++){
        for (uint j = 0; j < m->dim; j++){
            m->data[i][j] = 0;
        }
    }
}

void free_matrix(struct Matrix *m){
    // cout << "releasing matrix memory" << endl;
    if (m->data){
        for (uint i = 0; i < m->dim; i++){
            free(m->data[i]);
            // hipFree(m->data[i]);
        }
        free(m->data);
        // hipFree(m->data);
    }
}


void copy_matrix(struct Matrix *m, uint64_t * _data, uint source_dim){
    if (m->dim > 0 && m->data != NULL && m->dim != source_dim){
        // matrix dimension different - release memory first
        for (uint i = 0; i < m->dim; i++){
            free(m->data[i]);
            // hipFree(m->data[i]);
            // delete[] data[i];
        }
        free(m->data);
        // hipFree(m->data);
        // delete[] data;
    }
    if (m->dim != source_dim){
        m->data = (uint64_t **) malloc(sizeof(uint64_t*) * source_dim);
        // hipMallocManaged((void**) &m->data, sizeof(uint*) * source_dim);
        // data = new uint*[source_dim];
        for (uint i = 0; i < source_dim; i++){
            m->data[i] = (uint64_t *) malloc(sizeof(uint64_t) * source_dim);
            // hipMallocManaged((void**) &m->data[i], sizeof(uint) * source_dim);
            // data[i] = new uint[source_dim];
        }
    }
    m->dim = source_dim;
    for (uint i = 0; i < m->dim; i++){
        for (uint j = 0; j < m->dim; j++){
            m->data[i][j] = _data[i * source_dim + j];
        }
    }
}

void copy_matrix(struct Matrix *dst, struct Matrix * src){
    uint source_dim = src->dim;
    dst->unit = src->unit;
    dst->sdsm_info = src->sdsm_info;
    if (source_dim == 0){
        FLASHLOG("error when assigning an empty matrix");
        return;
    }
    if (dst->dim > 0 && dst->data != NULL && dst->dim != source_dim){
        // matrix dimension different - release memory first
        for (uint i = 0; i < dst->dim; i++){
            free(dst->data[i]);
            // hipFree(dst->data[i]);
        }
        free(dst->data);
        // hipFree(dst->data);
    }
    if (dst->dim != source_dim){
        dst->data = (uint64_t **) malloc(sizeof(uint64_t *) * source_dim);
        // hipMallocManaged((void**) &dst->data, sizeof(uint*) * source_dim);
        // data = new uint*[source_dim];
        for (uint i = 0; i < source_dim; i++){
            dst->data[i] = (uint64_t *) malloc(sizeof(uint64_t) * source_dim);
            // hipMallocManaged((void**) &dst->data[i], sizeof(uint) * source_dim);
            // data[i] = new uint[source_dim];
        }
    }
    dst->dim = source_dim;
    for (uint i = 0; i < dst->dim; i++){
        for (uint j = 0; j < dst->dim; j++){
            dst->data[i][j] = get_matrix(src, i, j);
        }
    }
}

bool equal_to_matrix(struct Matrix *a, struct Matrix *b){
    uint a_dim = a->dim, b_dim = b->dim;
    if  (a_dim != b_dim){
        return false;
    }
    for (uint i = 0; i < a_dim; i++){
        for (uint j = 0; j < a_dim; j++){
            if (a->data[i][j] != b->data[i][j]){
                return false;
            }
        }
    }
    return true;
}


uint64_t get_matrix(struct Matrix *m, uint x, uint y){
    if (x >= m->dim || y >= m->dim){
        FLASHLOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, m->dim, m->dim);
        return 0;
    }
    return m->data[x][y];
}

bool set_matrix(struct Matrix *m, uint64_t val, uint x, uint y){
    if (x >= m->dim || y >= m->dim){
        FLASHLOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, m->dim, m->dim);
        return false;
    }
    m->data[x][y] = val;
    return true;
}

bool add_matrix(struct Matrix *m, uint64_t val, uint x, uint y){
    if (x >= m->dim || y >= m->dim){
        FLASHLOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, m->dim, m->dim);
        return false;
    }
    m->data[x][y] += val;
    return true;
}

bool subtract_matrix(struct Matrix *m, uint64_t val, uint x, uint y){
    if (x >= m->dim || y >= m->dim){
        FLASHLOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, m->dim, m->dim);
        return false;
    }
    m->data[x][y] -= val;
    return true;
}

void scale_matrix(struct Matrix *m, uint factor){
    for (uint i = 0; i < m->dim; i++){
        for (uint j = 0; j < m->dim; j++){
            m->data[i][j] = (m->data[i][j] + factor - 1) / factor;
        }
    }
    m->unit = m->unit * factor;
}



void get_sdsm_info_matrix(struct Matrix *m){
    // reset sdsm info
    m->sdsm_info.is_sdsm = false;
    m->sdsm_info.non_max_row_n = 0;
    m->sdsm_info.non_max_col_n = 0;

    struct row_col_info_t max_row_idx[MAX_SERVER_NUM];
    struct row_col_info_t max_col_idx[MAX_SERVER_NUM];
    uint max_col_idx_n = 0, max_row_idx_n = 0;


    uint64_t max_sum = 0, same_sum_count = 0;
    for (uint i = 0; i < m->dim; i++){
        uint64_t row_sum = 0;
        for (uint j = 0; j < m->dim; j++){
            row_sum += m->data[i][j];
        }
        if (row_sum > max_sum){
            max_sum = row_sum;
            for (uint z = 0; z < max_row_idx_n; z++){
                m->sdsm_info.non_max_row[m->sdsm_info.non_max_row_n] = max_row_idx[z];
                m->sdsm_info.non_max_row_n ++;
            }
            max_row_idx[0].idx = i;
            max_row_idx[0].sum = row_sum;
            max_row_idx_n = 1;
        }else if (row_sum == max_sum){
            max_row_idx[max_row_idx_n].idx = i;
            max_row_idx[max_row_idx_n].sum = row_sum;
            max_row_idx_n ++;
            same_sum_count ++;
        }else{
            m->sdsm_info.non_max_row[m->sdsm_info.non_max_row_n].idx = i;
            m->sdsm_info.non_max_row[m->sdsm_info.non_max_row_n].sum = row_sum;
            m->sdsm_info.non_max_row_n ++;
        }
    }

    for (uint i = 0; i < m->dim; i++){
        uint64_t col_sum = 0;
        for (uint j = 0; j < m->dim; j++){
            col_sum += m->data[j][i];
        }
        if (col_sum > max_sum){
            max_sum = col_sum;
            for (uint z = 0; z < max_col_idx_n; z++){
                m->sdsm_info.non_max_col[m->sdsm_info.non_max_col_n] = max_col_idx[z];
                m->sdsm_info.non_max_col_n ++;
            }
            max_col_idx[0].idx = i;
            max_col_idx[0].sum = col_sum;
            max_col_idx_n = 1;

            for (uint z = 0; z < max_row_idx_n; z++){
                m->sdsm_info.non_max_row[m->sdsm_info.non_max_row_n] = max_row_idx[z];
                m->sdsm_info.non_max_row_n ++;
            }
            max_row_idx_n = 0;
        }else if (col_sum == max_sum){
            max_col_idx[max_col_idx_n].idx = i;
            max_col_idx[max_col_idx_n].sum = col_sum;
            max_col_idx_n ++;
            same_sum_count ++;
        }else{
            m->sdsm_info.non_max_col[m->sdsm_info.non_max_col_n].idx = i;
            m->sdsm_info.non_max_col[m->sdsm_info.non_max_col_n].sum = col_sum;
            m->sdsm_info.non_max_col_n ++;
        }
    }
    m->sdsm_info.is_sdsm = (same_sum_count == m->dim + m->dim);
    m->sdsm_info.max_row_col_sum = max_sum;
}

bool valid_matrix(struct Matrix *m) {
    return m->dim > 0 && m->data != NULL;
}

bool valid_sdsm_matrix(struct Matrix *m) {
    return m->dim > 0 && m->data != NULL && m->sdsm_info.is_sdsm;
}