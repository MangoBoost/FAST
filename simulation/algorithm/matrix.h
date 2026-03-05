#pragma once

#include <iostream>
#include <vector>
#include "define.h"
using namespace std;

struct row_col_info_t {
    uint idx;
    uint sum;
};

struct sdsm_info_t{
    bool is_sdsm;
    uint max_row_col_sum;
    vector<struct row_col_info_t> non_max_row;
    vector<struct row_col_info_t> non_max_col;
};


class Matrix {
private:
    uint ** data;
    uint dim;
    uint unit;
public:
    struct sdsm_info_t sdsm_info;
    Matrix(): data(NULL), dim(0), unit(1){sdsm_info.is_sdsm=false;sdsm_info.max_row_col_sum=0;};
    Matrix(uint _dim, uint _unit = 1);
    Matrix(uint** _data, uint _dim, uint _unit = 1);
    Matrix(uint* _data, uint _dim, uint _unit = 1);
    Matrix(Matrix * mat);
    ~Matrix();
    void copy(Matrix * mat);
    void copy(uint * _data, uint _dim);
    bool valid() {return dim > 0 && data != NULL;}
    bool valid_sdsm() {return dim > 0 && data != NULL && sdsm_info.is_sdsm;}
    uint get_dim() {return dim;}
    uint get_unit() {return unit;}
    uint get(uint x, uint y);
    bool set(uint val, uint x, uint y);
    bool add(uint val, uint x, uint y);
    bool subtract(uint val, uint x, uint y);
    void scale(uint factor);
    bool equal_to(Matrix * mat);
    void get_sdsm_info();
    void print();
};