#include "all2all.h"
#include "matrix.h"
#include "define.h"
#include <iomanip>
#include <pthread.h>
using namespace std;


FastAll2All::FastAll2All(Matrix * _mat, uint _gpu_n, INTER_LINK_TYPE _type, INTRA_LINK_TYPE _itype, double MBpu){
    is_scaled = false;
    mat.copy(_mat);
    uint dim = mat.get_dim();
    hungarian_info.matching.insert(hungarian_info.matching.end(), dim*2, -1);
    hungarian_info.visit.insert(hungarian_info.visit.end(), dim*2, false);
    unordered_set<uint> empty_vector;
    hungarian_info.row_to_col.insert(hungarian_info.row_to_col.end(), dim, empty_vector);
    MB_per_unit = MBpu;
    inter_type = _type;
    inter_link = get_inter_link_info(_type);
    intra_type = _itype;
    intra_link = get_intra_link_info(_itype);
    scaling_factor = 1;
    gpu_n = _gpu_n;
}

void FastAll2All::to_scaled_doubly_stochastic_matrix(){
    if (!is_scaled){
        mat.get_sdsm_info();
        SDS_mat.copy(&mat);
    }else{
        S_mat.get_sdsm_info();
        // cout << "scaling factor " << scaling_factor  << ", max row/col sum: " << S_mat.sdsm_info.max_row_col_sum << endl;
        SDS_mat.copy(&S_mat);
    }

    if(!SDS_mat.sdsm_info.is_sdsm){
        uint dim = SDS_mat.get_dim();
        uint max_sum = SDS_mat.sdsm_info.max_row_col_sum;
        // original matrix is not SDSM, do the conversion
        for (vector<struct row_col_info_t>::iterator row = SDS_mat.sdsm_info.non_max_row.begin(); row != SDS_mat.sdsm_info.non_max_row.end(); row++){
            for (vector<struct row_col_info_t>::iterator col = SDS_mat.sdsm_info.non_max_col.begin(); col != SDS_mat.sdsm_info.non_max_col.end(); col++){
                if (col -> sum == max_sum)
                    continue;
                uint diff =  max_sum - MAX(row -> sum, col -> sum);
                SDS_mat.add(diff, row -> idx, col -> idx);
                row -> sum += diff;
                col -> sum += diff;
                if (row -> sum == max_sum) {
                    break;
                }          
            }
        }
        SDS_mat.sdsm_info.is_sdsm = true;
        SDS_mat.sdsm_info.non_max_row.clear();
        SDS_mat.sdsm_info.non_max_col.clear();
        // SDS_mat.get_sdsm_info();
    }
}

void FastAll2All::decompose(){
    if (!SDS_mat.valid_sdsm()){
        LOG("error when doing decomposition, must convert matrix to sdsm first!");
        return;
    }
    p_sets.clear(); // store results
    uint freq_sum = 0, max_sum = SDS_mat.sdsm_info.max_row_col_sum;
    while(freq_sum < max_sum){
        update_edges();
        hungarian();
        freq_sum += update_permutation_sets();
    }
}


uint FastAll2All::hungarian(){
    uint match_num = 0;
    uint dim = SDS_mat.get_dim();
    std::fill(hungarian_info.matching.begin(), hungarian_info.matching.end(), -1);
    for (uint u = 0; u < dim; u++){
        if (hungarian_info.matching[u] == -1){
            std::fill(hungarian_info.visit.begin(), hungarian_info.visit.end(), false);
            if(hungarian_dfs(u))
                match_num ++;
        }
    }
    return match_num;
}


bool FastAll2All::hungarian_dfs(uint u){
    for (unordered_set<uint>::iterator col_idx = hungarian_info.row_to_col[u].begin(); col_idx != hungarian_info.row_to_col[u].end(); col_idx ++){
        if (!hungarian_info.visit[*col_idx]){
            hungarian_info.visit[*col_idx] = true;
            if (hungarian_info.matching[*col_idx] == -1 || hungarian_dfs(hungarian_info.matching[*col_idx])){
                hungarian_info.matching[*col_idx] = u;
                hungarian_info.matching[u] = *col_idx;
                return true;
            }
        }
    }
    return false;
}

void FastAll2All::update_edges(){
    uint dim = SDS_mat.get_dim();
    // row vertices id: 0 - dim-1, col vertices id: dim - 2*dim-1
    for (uint i = 0; i < dim; i++){
        for (uint j = 0; j < dim; j++){
            uint col_id = j + dim;
            if (SDS_mat.get(i, j) > 0 ){
                hungarian_info.row_to_col[i].insert(col_id);
            }else if (SDS_mat.get(i, j) == 0){
                hungarian_info.row_to_col[i].erase(col_id);
            }
        }
    }
}

uint FastAll2All::update_permutation_sets(){
    uint dim = SDS_mat.get_dim();
    // row vertices id: 0 - dim-1, col vertices id: 0 - dim-1
    PermutationSet r(1, is_scaled ? scaling_factor : 1, dim);
    uint min_freq = SDS_mat.get(0, hungarian_info.matching[0] - dim);
    for(uint i = 0; i < dim; i++){
        uint col_id = hungarian_info.matching[i] - dim;
        min_freq = MIN(SDS_mat.get(i, col_id), min_freq);
        r.mp.insert(make_pair(i, col_id));
    }

    for(uint i = 0; i < dim; i++){
        uint col_id = hungarian_info.matching[i] - dim;
        SDS_mat.subtract(min_freq, i, col_id);
    }

    r.set_freq(min_freq);
    p_sets.push_back(r);
    return min_freq;
}

void FastAll2All::print_decomposition(){
    for (vector<PermutationSet>::iterator ps = p_sets.begin(); ps != p_sets.end(); ps++){
        ps -> print_permutation_matrix();
    }
}

bool FastAll2All::verify_decomposition(){
    uint dim = SDS_mat.get_dim();
    Matrix r(dim);
    for (vector<PermutationSet>::iterator ps = p_sets.begin(); ps != p_sets.end(); ps++){
        for (uint i = 0; i < dim; i++){
            uint non_empty_col_id = ps -> mp[i];
            r.add(ps->get_freq()/scaling_factor, i, non_empty_col_id);
        }
    }
    r.get_sdsm_info();
    to_scaled_doubly_stochastic_matrix();
    return SDS_mat.equal_to(&r);
}

void PermutationSet::print_permutation_matrix(){
    cout << "Permutation Matrix:" << endl;
    for(uint i = 0; i < dim; i ++){
        uint non_empty_col_id = mp[i];
        for (uint j = 0; j < dim; j++){
            cout << setw(10);
            if (non_empty_col_id == j){
                cout << frequency * scaling_factor;
            }else{
                cout << "0";
            }
        }
        cout << endl;
     }
}
struct scale_info_t FastAll2All::scale_matrix(uint factor, uint* buffer){
    uint dim = mat.get_dim();
    for (uint i = 0; i < dim; i ++){
        for (uint j = 0; j < dim; j++){
            uint idx = i * dim + j;
            buffer[idx] = (mat.get(i, j) + factor - 1) / factor;    // ceiling division 
        }
    }
    // collect information after scaling
    uint max_sum = 0;
    for (uint i = 0; i < dim; i++){
        uint row_sum = 0, col_sum = 0;
        for (uint j = 0; j < dim; j++){
            row_sum += buffer[i * dim + j];
            col_sum += buffer[j * dim + i];
        }
        max_sum = MAX(max_sum, MAX(row_sum, col_sum));
    }
    struct scale_info_t r = {.max_sum = max_sum, .factor=factor};
    return r;
}

bool limit_max_sum(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t link, struct link_info_t intra_link, double MBpu){
    return (s_info.max_sum <= MAX_SUM_LIMIT);
}

bool always_scale(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu){
    if (s_info.max_sum <= server_n){
        return true;
    }
    return false;
}

bool balance_alpha_beta(uint server_n, uint gpu_n, struct scale_info_t s_info, struct link_info_t inter_link, struct link_info_t intra_link, double MBpu){
    if (s_info.max_sum <= server_n){
        return true;
    }
    double alpha = inter_link.alpha * server_n * server_n + intra_link.alpha * server_n * server_n * gpu_n;
    double beta = inter_link.beta * s_info.max_sum * s_info.factor * MBpu;
    // cout << "factor: " << s_info.factor << ", alpha: " << alpha << ", beta: " << beta << " , MBpu: " << MBpu << endl;
    return alpha/beta < 0.1;
    // cost of spreadout: m*n-1 inter-transfer
    // cost of proposed algorithm: n*n inter-transfer, n*n*m intra-transfer
    // cost of proposed algorithm after making each element to one: n inter-transfer, n*m intra transfer
}

//return scaling factor that satisfies the condition, -1 means not finding a valid scaling factor
int FastAll2All::scaling_binary_search(scaling_success_condition fn, int l, int r, uint step, uint* buffer){
    if (l > r){
        return -1;
    }
    uint ele_num = (r - l) / step + 1;
    uint mid = l + ele_num / 2 * step;
    struct scale_info_t scale_result = scale_matrix(mid, buffer);
    uint dim = mat.get_dim();

    if ((*fn)(dim, gpu_n, scale_result, inter_link, intra_link, MB_per_unit)) {
        // mid satisfies the condition, try to find a smaller factor
        int ret = scaling_binary_search(fn, l, mid - step, step, buffer);
        return (ret == -1) ? mid : ret;
    }else{
        // mid does not work, try a bigger factor
        return scaling_binary_search(fn, mid + step, r, step, buffer);
    }
}


void FastAll2All::to_scaled_matrix(scaling_success_condition fn){
    is_scaled = false;
    uint dim = mat.get_dim();
    uint * sm = new uint[dim * dim];
    int ret = scaling_binary_search(fn, SCALING_FACTOR_MIN, SCALING_FACTOR_MAX, SCALING_FACTOR_STEP, sm);
    // cout<<"factor: "<<ret << endl;
    delete[] sm;
    if (ret == -1){
        cout << "no scaling factor can satisfy the condition, change search range!" <<endl;
    }else{
        S_mat.copy(&mat);
        S_mat.scale(ret);
        is_scaled = true;
        scaling_factor = ret;
    }
}

