#include "fast_alltoall/alltoall_algorithm.h"
#include "fast_alltoall/alltoall_matrix.h"
#include "fast_alltoall/alltoall_define.h"
#include <iomanip>


void init_fastall2all(struct FastAll2All * ata, Matrix * _mat){
    init_matrix(&(ata->mat));
    init_matrix(&(ata->SDS_mat));
    copy_matrix(&(ata->mat), _mat);
    uint dim = ata->mat.dim;
    for (uint i = 0; i < dim * 2; i ++){
        ata->hungarian_info.matching[i] = -1;
        ata->hungarian_info.visit[i] = false;
    }
    for (uint i = 0; i < dim; i++){
        ata->hungarian_info.row_to_col_n[i] = 0;
    }
    ata->p_sets_n = 0;

}

void free_fastall2all(struct FastAll2All * ata){
    free_matrix(&(ata->mat));
    free_matrix(&(ata->SDS_mat));
}


void to_scaled_doubly_stochastic_matrix_fastall2all(struct FastAll2All * ata){
    get_sdsm_info_matrix(&(ata->mat));
    copy_matrix(&(ata->SDS_mat), &(ata->mat));

    if(!ata->SDS_mat.sdsm_info.is_sdsm){
        uint dim = ata->SDS_mat.dim;
        uint64_t max_sum = ata->SDS_mat.sdsm_info.max_row_col_sum;
        // original matrix is not SDSM, do the conversion
        for (uint row_id = 0; row_id < ata->SDS_mat.sdsm_info.non_max_row_n; row_id++){
            struct row_col_info_t * row = &(ata->SDS_mat.sdsm_info.non_max_row[row_id]);
            for (uint col_id = 0; col_id < ata->SDS_mat.sdsm_info.non_max_col_n; col_id++){
                struct row_col_info_t * col = &(ata->SDS_mat.sdsm_info.non_max_col[col_id]);
                if (col -> sum == max_sum){
                    continue;
                }
                uint64_t diff = max_sum - MAX(row -> sum, col -> sum);
                add_matrix(&ata -> SDS_mat, diff, row -> idx, col -> idx);
                row -> sum += diff;
                col -> sum += diff;
                if (row -> sum == max_sum){
                    break;
                }
            }
        }
        ata->SDS_mat.sdsm_info.is_sdsm = true;
        ata->SDS_mat.sdsm_info.non_max_row_n = 0;
        ata->SDS_mat.sdsm_info.non_max_col_n = 0;
    }
}

void sort_permutation_sets(struct FastAll2All * ata){
    for (uint i = 0; i < ata->p_sets_n; i++){
        ata -> p_sets_ascending[i] = i;
    }
    for (uint i = 1; i < ata->p_sets_n; i++){
        uint idx = ata -> p_sets_ascending[i];
        uint64_t freq = ata->p_sets[idx].frequency;
        int j = i - 1;
        while (j >= 0 && ata->p_sets[ata -> p_sets_ascending[j]].frequency > freq){
            ata -> p_sets_ascending[j + 1] = ata -> p_sets_ascending[j];
            j --;
        }
        ata -> p_sets_ascending[j + 1] = idx;
    }
    // std::cout<<"[SORT]: " << std::endl;
    // for (uint i = 0; i < ata->p_sets_n; i++){
    //     std::cout << ata->p_sets[ata -> p_sets_ascending[i]].frequency << " ";
    // }
    // std::cout<<std::endl;
}

void decompose_fastall2all(struct FastAll2All * ata){
    if (!valid_sdsm_matrix(&ata->SDS_mat)){
        FLASHLOG("error when doing decomposition, must convert matrix to sdsm first!");
        return;
    }
    ata->p_sets_n = 0;
    uint64_t freq_sum = 0, max_sum = ata->SDS_mat.sdsm_info.max_row_col_sum;
    while(freq_sum < max_sum){
        update_edges_fastall2all(ata);
        hungarian_fastall2all(ata);
        freq_sum += update_permutation_sets_fastall2all(ata);
    }
    sort_permutation_sets(ata);
}


uint hungarian_fastall2all(struct FastAll2All * ata){
    uint match_num = 0;
    uint dim = ata->SDS_mat.dim;
    for (uint i = 0; i < dim * 2; i++){
        ata->hungarian_info.matching[i] = -1;
    }
    for (uint u = 0; u < dim; u++){
        if (ata->hungarian_info.matching[u] == -1){
            for (uint i = 0; i < dim * 2; i++){
                ata->hungarian_info.visit[i] = false;
            }
            if(hungarian_dfs_fastall2all(ata, u))
                match_num ++;
        }
    }
    return match_num;
}


bool hungarian_dfs_fastall2all(struct FastAll2All * ata, uint u){
    for (uint i = 0; i < ata->hungarian_info.row_to_col_n[u]; i++){
        uint col =  ata->hungarian_info.row_to_col[u][i];
        if (!ata->hungarian_info.visit[col]){
            ata->hungarian_info.visit[col] = true;
            if (ata->hungarian_info.matching[col] == -1 || hungarian_dfs_fastall2all(ata, ata->hungarian_info.matching[col])){
                ata->hungarian_info.matching[col] = u;
                ata->hungarian_info.matching[u] = col;
                return true;
            }
        }
    }
    return false;
}


void set_insert(uint val, uint * array, uint * sz){
    for (uint i = 0; i < (*sz); i ++){
        if (array[i] == val){
            return;
        }
    }
    array[(*sz)] = val;
    (*sz) ++;
}

void set_remove(uint val, uint * array, uint * sz){
    if (*sz == 0) return;
    uint i = 0;
    for (i = 0; i < (*sz); i++){
        if(array[i] == val){
            break;
        }
    }
    for (uint j = i; j < (*sz) - 1; j++){
        array[j] = array[j+1];
    }
    if (i < (*sz)){
        (*sz) --;
    }
}


void update_edges_fastall2all(struct FastAll2All * ata){
    uint dim = ata->SDS_mat.dim;
    // row vertices id: 0 - dim-1, col vertices id: dim - 2*dim-1
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            uint col_id = j + dim;
            uint64_t elem = get_matrix(&ata ->SDS_mat, i, j);
            if(elem > 0){
                // insert edge into the set
                set_insert(col_id, ata->hungarian_info.row_to_col[i], &ata->hungarian_info.row_to_col_n[i]);
            }else if (elem == 0){
                // remove edge from set
                set_remove(col_id, ata->hungarian_info.row_to_col[i], &ata->hungarian_info.row_to_col_n[i]);
            }
        }
    }
}

uint update_permutation_sets_fastall2all(struct FastAll2All * ata){
    uint dim = ata->SDS_mat.dim;
    // row vertices id: 0 - dim-1, col vertices id: 0 - dim-1

    struct PermutationSet * cur_pset = &ata->p_sets[ata->p_sets_n];
    init_permutation_set(cur_pset, 1, 1, dim);
    uint64_t min_freq = get_matrix(&ata->SDS_mat, 0, ata->hungarian_info.matching[0] - dim);
    for(uint i = 0; i < dim; i++){
        uint col_id = ata->hungarian_info.matching[i] - dim;
        min_freq = MIN(get_matrix(&ata->SDS_mat, i, col_id), min_freq);
        map_insert(cur_pset -> mp, & cur_pset -> mp_n, i, col_id);
    }

    for(uint i = 0; i < dim; i++){
        uint col_id = ata->hungarian_info.matching[i] - dim;
        subtract_matrix(&ata->SDS_mat, min_freq, i, col_id);
    }

    set_freq_permutation_set(cur_pset, min_freq);
    ata->p_sets_n ++;
    return min_freq;
}

void print_decomposition_fastall2all(struct FastAll2All * ata){
    for (uint z = 0; z < ata->p_sets_n; z++){
        print_permutation_set(&ata->p_sets[z]);
    }
}

bool verify_decomposition_fastall2all(struct FastAll2All * ata){
    uint dim = ata->SDS_mat.dim;
    struct Matrix r;
    init_matrix(&r, dim);
    for (uint z = 0; z < ata->p_sets_n; z ++){
        for (uint i = 0; i < dim; i++){
            uint non_empty_col_id;
            if (map_lookup(ata->p_sets[z].mp, ata->p_sets[z].mp_n, i, &non_empty_col_id)){
                add_matrix(&r, ata->p_sets[z].frequency, i, non_empty_col_id);
            }else{
                FLASHLOG("map error lookup");
                exit(1);
            }
        }
    }
    get_sdsm_info_matrix(&r);
    to_scaled_doubly_stochastic_matrix_fastall2all(ata);
    return equal_to_matrix(&ata->SDS_mat, &r);
}

void print_permutation_set(struct PermutationSet * ps){
    cout << "Permutation Set, dim: " << ps->dim << endl;
    for(uint i = 0; i < ps->dim; i ++){
        uint non_empty_col_id;
        if (map_lookup(ps->mp, ps->mp_n, i, &non_empty_col_id)){
            for (uint j = 0; j < ps->dim; j++){
                cout << setw(10);
                if (non_empty_col_id == j){
                    cout << ps->frequency * ps->scaling_factor;
                }else{
                    cout << "0";
                }
            }
            cout << endl;
        }else{
            FLASHLOG("map error lookup");
            exit(1);
        }

     }
}

void to_server_permutation_set(struct PermutationSet * ps, uint server_n, uint * r){
    for (uint i = 0; i < server_n; i++){
        uint non_empty_col_id;
        if (map_lookup(ps->mp, ps->mp_n, i, &non_empty_col_id)){
            r[i] = non_empty_col_id;
        }else{
            FLASHLOG("map error lookup");
            exit(1);
        }
    }
}

void from_server_permutation_set(struct PermutationSet * ps, uint server_n, uint * r){
    for (uint i = 0; i < server_n; i++){
        uint non_empty_col_id;
        if (map_lookup(ps->mp, ps->mp_n, i, &non_empty_col_id)){
            r[non_empty_col_id] = i;
        }else{
            FLASHLOG("map error lookup");
            exit(1);
        }
    }
}

void init_permutation_set(struct PermutationSet * ps, uint64_t _freq, uint _sf, uint _dim) {
    ps->frequency = _freq;
    ps->scaling_factor = _sf;
    ps->dim = _dim;
    ps->mp_n = 0;
}


void map_insert(struct map_data_t * array, uint * sz, uint key, uint val){
    array[(*sz)].key = key;
    array[(*sz)].val = val;
    (*sz) ++;
}

bool map_lookup(struct map_data_t * array, uint sz, uint key, uint * val){
    for (uint i = 0; i < sz; i++){
        if (array[i].key == key){
            (*val) = array[i].val;
            return true;
        }
    }
    return false;
}

void set_freq_permutation_set(struct PermutationSet * ps, uint64_t freq){
    ps -> frequency = freq;
}

uint64_t get_freq_permutation_set(struct PermutationSet * ps){
    return ps -> frequency * ps -> scaling_factor;
}