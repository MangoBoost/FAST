#include <iostream>
#include <stdio.h>
#include <iomanip>
#include "matrix.h"

using namespace std;

Matrix::Matrix(uint _dim, uint _unit){
    unit = _unit;
    dim = _dim;
    sdsm_info.max_row_col_sum = 0;
    sdsm_info.is_sdsm = false;
    data = new uint*[_dim];
    for (uint i = 0; i < _dim; i++){
        data[i] = new uint[_dim];
        for (uint j = 0; j < _dim; j++){
            data[i][j] = 0;
        }
    }
}

Matrix::Matrix(uint* _data, uint _dim, uint _unit){
    unit = _unit;
    dim = _dim;
    sdsm_info.max_row_col_sum = 0;
    sdsm_info.is_sdsm = false;
    data = new uint*[_dim];
    uint idx = 0;
    for (uint i = 0; i < _dim; i ++){
        data[i] = new uint[_dim];
        for (uint j = 0; j < _dim; j++){
            data[i][j] = _data[idx];
            idx ++;
        }
    }
}

Matrix::Matrix(uint** _data, uint _dim, uint _unit){
    unit = _unit;
    dim = _dim;
    sdsm_info.max_row_col_sum = 0;
    sdsm_info.is_sdsm = false;
    data = new uint*[_dim];
    for (uint i = 0; i < _dim; i ++){
        data[i] = new uint[_dim];
        for (uint j = 0; j < _dim; j++){
            data[i][j] = _data[i][j];
        }
    }
}


// hard copy - initiate a matrix with an exisiting matrix
Matrix::Matrix(Matrix * mat){
    dim = mat->get_dim();
    unit = mat->get_unit();
    sdsm_info = mat->sdsm_info;
    data = new uint*[dim];
    for (uint i = 0; i < dim; i++){
        data[i] = new uint[dim];
        for (uint j = 0; j < dim; j++){
            data[i][j] = mat->get(i,j);
        }
    }
}

// hard copy
void Matrix::copy(Matrix * mat){
    uint source_dim = mat->get_dim();
    unit = mat->get_unit();
    sdsm_info = mat->sdsm_info;
    if (source_dim == 0){
        LOG("error when assigning an empty matrix");
        return;
    }
    if (dim > 0 && data != NULL && dim != source_dim){
        // matrix dimension different - release memory first
        for (uint i = 0; i < dim; i++){
            delete[] data[i];
        }
        delete[] data;
    }
    if (dim != source_dim){
        data = new uint*[source_dim];
        for (uint i = 0; i < source_dim; i++){
            data[i] = new uint[source_dim];
        }
    }
    dim = source_dim;
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            data[i][j] = mat->get(i,j);
        }
    }
}

void Matrix::copy(uint * _data, uint source_dim){
    if (dim > 0 && data != NULL && dim != source_dim){
        // matrix dimension different - release memory first
        for (uint i = 0; i < dim; i++){
            delete[] data[i];
        }
        delete[] data;
    }
    if (dim != source_dim){
        data = new uint*[source_dim];
        for (uint i = 0; i < source_dim; i++){
            data[i] = new uint[source_dim];
        }
    }
    dim = source_dim;
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            data[i][j] = _data[i * source_dim + j];
        }
    }
}


bool Matrix::equal_to(Matrix * mat){
    uint source_dim = mat->get_dim();
    if  (dim != source_dim){
        return false;
    }
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            if (data[i][j] != mat->get(i,j)){
                return false;
            }
        }
    }
    return true;
}


Matrix::~Matrix(){
    // cout << "releasing matrix memory" << endl;
    for (uint i = 0; i < dim; i++){
        delete[] data[i];
    }
    delete[] data;
}

uint Matrix::get(uint x, uint y){
    if (x >= dim || y >= dim){
        LOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, dim, dim);
        return 0;
    }
    return data[x][y];
}

bool Matrix::set(uint val, uint x, uint y){
    if (x >= dim || y >= dim){
        LOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, dim, dim);
        return false;
    }
    data[x][y] = val;
    return true;
}

bool Matrix::add(uint val, uint x, uint y){
    if (x >= dim || y >= dim){
        LOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, dim, dim);
        return false;
    }
    data[x][y] += val;
    return true;
}

bool Matrix::subtract(uint val, uint x, uint y){
    if (x >= dim || y >= dim){
        LOG("invalid get parameters (%u, %u) for a %u x %u matrix", x, y, dim, dim);
        return false;
    }
    data[x][y] -= val;
    return true;
}

void Matrix::scale(uint factor){
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            data[i][j] = (data[i][j] + factor - 1) / factor;
        }
    }
    unit = unit * factor;
}

void Matrix::print(){
    printf("Print %u x %u matrix:\n", dim, dim);
    for(uint i = 0; i < dim; i ++){
        for (uint j = 0; j < dim; j++){
            cout << setw(10);
            cout << data[i][j];
        }
        cout << endl;
    }
    cout << endl << "is_sdsm: "<< (sdsm_info.is_sdsm ? "true" : "false") << ", max_row_col_sum: " << sdsm_info.max_row_col_sum << endl;
    cout << "non_max_row_idx: ";
    for (vector<struct row_col_info_t>::iterator row = sdsm_info.non_max_row.begin(); row != sdsm_info.non_max_row.end(); row++){
        cout << row->idx << " ";
    }
    cout << endl << "non_max_col_idx: ";
    for (vector<struct row_col_info_t>::iterator col = sdsm_info.non_max_col.begin(); col != sdsm_info.non_max_col.end(); col++){
        cout << col->idx << " ";
    }
    cout << endl;
}

void Matrix::get_sdsm_info(){
    // reset sdsm info
    sdsm_info.is_sdsm = false;
    sdsm_info.non_max_row.clear();
    sdsm_info.non_max_col.clear();
    vector<struct row_col_info_t> max_row_idx;
    vector<struct row_col_info_t> max_col_idx;
    uint max_sum = 0, same_sum_count = 0;
    for (uint i = 0; i < dim; i++){
        uint row_sum = 0;
        for (uint j = 0; j < dim; j++){
            row_sum += data[i][j];
        }
        struct row_col_info_t row_temp = {.idx = i, .sum = row_sum};
        if (row_sum > max_sum){
            max_sum = row_sum;
            std::copy(make_move_iterator(max_row_idx.begin()), make_move_iterator(max_row_idx.end()), back_inserter(sdsm_info.non_max_row));
            max_row_idx.clear();
            max_row_idx.push_back(row_temp);
        }else if (row_sum == max_sum){
            max_row_idx.push_back(row_temp);
            same_sum_count ++;
        }else{
            sdsm_info.non_max_row.push_back(row_temp);
        }
    }

    for (uint i = 0; i < dim; i++){
        uint col_sum = 0;
        for (uint j = 0; j < dim; j++){
            col_sum += data[j][i];
        }
        struct row_col_info_t col_temp = {.idx = i, .sum = col_sum};
        if (col_sum > max_sum){
            max_sum = col_sum;
            std::copy(make_move_iterator(max_col_idx.begin()), make_move_iterator(max_col_idx.end()), back_inserter(sdsm_info.non_max_col));
            max_col_idx.clear();
            max_col_idx.push_back(col_temp);
            std::copy(make_move_iterator(max_row_idx.begin()), make_move_iterator(max_row_idx.end()), back_inserter(sdsm_info.non_max_row));
            max_row_idx.clear();
        }else if (col_sum == max_sum){
            max_col_idx.push_back(col_temp);
            same_sum_count ++;
        }else{
            sdsm_info.non_max_col.push_back(col_temp);
        }
    }
    sdsm_info.is_sdsm = (same_sum_count == dim + dim);
    sdsm_info.max_row_col_sum = max_sum;
}
