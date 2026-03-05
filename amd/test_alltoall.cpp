#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fast_alltoall/alltoall_global_scheduler.h>
#include <fast_alltoall/alltoall_local_scheduler.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <ctime>
#include <nccl.h>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

using namespace std::chrono;
using namespace std;

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("error when executing cmd:%u",cmd);      \
    exit(1);                                      \
  }                                                 \
} while(0)


#define RCCLCHECK(cmd) do {                         \
  hipError_t res = cmd;                           \
  if (res != hipSuccess) {                         \
    printf("error when executing cmd: %u, %s\n",cmd,      \
            hipGetErrorString(res));                   \
    exit(1);                                      \
  }                                                 \
} while(0)



uint64_t zipf_inverse_cdf_fast(double s, double p, uint64_t N){
    if (p > 1.0 || p < 0){
        printf("ERROR: probability must be within [0,1]\n");
        return 0.0;
    }

    double tolerance = 0.01;
    double x = (double) N / 2.0;

    double D = p * (12 * (pow(N, 1 - s) - 1) / (1 - s) +
                    6 - 6 * pow(N, -s) +
                    s - pow(N, -1 - s) * s);
    while (1){
        double m = pow(x, -2 - s);
        double mx = m * x;
        double mxx = mx * x;
        double mxxx = mxx * x;

        double a = 12 * (mxxx - 1) / (1 - s) + 6 * (1 - mxx) + (s - (mx * s)) - D;
        double b = 12 * mxx + 6 * (s * mx) + (m * s * (s + 1));
        double newx = MAX(1, x - a / b);
        if (fabs(newx - x) <= tolerance)
            return uint64_t(newx);
        x = newx;
    }
}


void zipf_distribution(uint64_t * workload, double s, uint nrank, uint64_t bound){
    double p = 0.0;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
             p = (double) rand() / (double) RAND_MAX;
            workload[i * nrank + j] = mem_align(zipf_inverse_cdf_fast(s, p, bound));
        }
    }
    // clean the diagnal
    for (uint i = 0; i < nrank; i++){
         workload[i * nrank + i] = 0;
    }
}

void zipf_distribution2(uint64_t * workload, double s, uint nrank, uint64_t per_gpu_sz){
    uint64_t total_sz = 0;
    double p = 0.0;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            if (i == j){
                workload[i * nrank + j] = 0;
                continue;
            }
            p = (double) rand() / (double) RAND_MAX;
            workload[i * nrank + j] = zipf_inverse_cdf_fast(s, p, 1024);
            total_sz += workload[i * nrank + j];
        }
    }
    uint64_t multiplier = per_gpu_sz * (nrank - 1) * nrank / total_sz;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = mem_align(workload[i * nrank + j] * multiplier);
        }
    }
}

void uniform_distribution(uint64_t * workload, uint nrank, uint64_t bound){
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = mem_align(rand() % bound);
        }
    }
    // clean the diagnal
    for (uint i = 0; i < nrank; i++){
         workload[i * nrank + i] = 0;
    }
}

void uniform_distribution2(uint64_t * workload, uint nrank, uint64_t per_gpu_sz){
    uint64_t total_sz = 0;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            if (i == j) {
                workload[i * nrank + j] = 0;
                continue;
            }
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = rand() % 1024;
            total_sz += workload[i * nrank + j];
        }
    }
    uint64_t multiplier = per_gpu_sz * (nrank - 1) * nrank / total_sz;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = mem_align(workload[i * nrank + j] * multiplier);
        }
    }
}

void fixed_distribution(uint64_t * workload, uint nrank, uint64_t bound){
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            workload[i * nrank + j] = mem_align(bound);
            // workload[i * nrank + j] = (rand() % bound + 0x1ff) & 0xfffffe00;
        }
    }
    // clean the diagnal
    for (uint i = 0; i < nrank; i++){
         workload[i * nrank + i] = 0;
    }
}



struct alltoall_parameters{
    void * sendbuff;
    void * recvbuff;
    void * tempbuff;
    void * verifybuff;
    size_t * sendcount;
    size_t * sendpos;
    size_t * recvcount;
    size_t * recvpos;
    struct scheduling_result_t * sched;
};


struct buffer_ptr_t{
    void * sendbuff;
    void * recvbuff;
    void * lbsend;
    void * lbrecv;
    void * crosbuff1;
    void * crosbuff2;
    void * rstrbuff;
    void * verifybuff;
};


void print_sendbuffs(void * send_buff, uint MAX_BUFFER_SIZE_PER_RANK, uint dim, uint gpu_n, uint rank){
    std::cout << "send buffs: BUFFER_SIZE_PER_RANK " << MAX_BUFFER_SIZE_PER_RANK << ", rank " << rank << ", dim: " << dim << std::endl;
    for (uint j = 0; j < dim; j++){
        for (uint k = 0; k < gpu_n; k++){
            for (uint z = 0; z < MAX_BUFFER_SIZE_PER_RANK; z++){
                int32_t * ptr = (int32_t *) send_buff + j * gpu_n * MAX_BUFFER_SIZE_PER_RANK + k * MAX_BUFFER_SIZE_PER_RANK + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << "|";
        }
        std::cout << "  ";
    }
    std::cout << dec << std::endl;

}

void print_recvbuffs(void * recv_buff, uint MAX_BUFFER_SIZE_PER_RANK, uint dim, uint rank){
    std::cout << "recv buffs: BUFFER_SIZE_PER_RANK " << MAX_BUFFER_SIZE_PER_RANK << ", rank " << rank << ", dim: " << dim << std::endl;
    for (uint j = 0; j < dim; j++){
        for (uint z = 0; z < MAX_BUFFER_SIZE_PER_RANK; z++){
            int32_t * ptr = (int32_t *) recv_buff + j * MAX_BUFFER_SIZE_PER_RANK + z;
            std::cout << hex << std::setfill('0') << std::setw(8)<< *ptr;
        }
        std::cout << " ";
    }
    std::cout << dec << std::endl;
}

void analyze_buffs(struct buffer_ptr_t * bufs, struct buffer_parameter_t * buff_parameter, uint rank, uint server_n, uint gpu_n){
    std::cout << "----------rank " << rank << "-----------" << "correct: " <<  (0 == memcmp((char*)bufs->recvbuff, (char*)bufs->verifybuff,  660 * sizeof(int32_t)))<<std::endl;
    uint total_sz = 0;
    for (uint i = 0; i < server_n * gpu_n; i++){
        char* recvbuff = (char*)bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        char* verifybuff =  (char*) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        uint sz = buff_parameter -> recvbuff_sz[i];
        std::cout << "from rank " << i << " sz: " << sz << std::endl;
        total_sz += sz;
        if(0 != memcmp(recvbuff, verifybuff,  sz*sizeof(int32_t))){
            std::cout << "from rank " << i << " buffer error" << std::endl;
            std::cout<< "recvbuff:" <<std::endl;
            for (uint z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
            std::cout<< "verifybuff:" <<std::endl;
            for (uint z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
        }

    }
    std::cout << "total sz: " << total_sz << std::endl;
}


void print_buffs(struct buffer_ptr_t * bufs, struct buffer_parameter_t * buff_parameter, uint rank, uint server_n, uint gpu_n){

    std::cout << "----------rank " << rank << "-----------" << std::endl;

    std::cout << "sendbuff: " << std::endl;
    uint dim = server_n * gpu_n;
    for (uint j = 0; j < dim; j++){
        for (uint k = 0; k < gpu_n; k++){
            uint sz = buff_parameter -> sendbuff_region[j].src_gpu_sz[k];
            for (uint z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->sendbuff + buff_parameter -> sendbuff_region[j].src_gpu_disp[k] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << "|";
        }
        std::cout << "  ";
    }

    std::cout << std::endl << "lbsend: " << std::endl;

    for (uint i = 0; i < gpu_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            for (uint s = 0; s < server_n; s++){
                uint sz = buff_parameter -> lbsend_area[i].dst_gpu_region[j].server_sz[s];
                for (uint z = 0; z < sz; z ++){
                    int32_t * ptr = (int32_t *) bufs->lbsend + buff_parameter -> lbsend_area[i].dst_gpu_region[j].server_disp[s] + z;
                    std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
                }
                std::cout << "+";
            }
            std::cout << "|";
        }
        std::cout << " ";
    }

    std::cout << std::endl << "recvbuff: " << std::endl;

    for (uint j = 0; j < dim; j++){
        uint sz = buff_parameter -> recvbuff_sz[j];
        for (uint z = 0; z < sz; z++){
            int32_t * ptr = (int32_t *) bufs->recvbuff +  buff_parameter -> recvbuff_disp[j] + z;
            std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
        }
        std::cout << "  ";
    }

    std::cout << std::endl << "verifybuff: " << std::endl;
    for (uint j = 0; j < dim; j++){
        uint sz = buff_parameter -> recvbuff_sz[j];
        for (uint z = 0; z < sz; z++){
            int32_t * ptr = (int32_t *) bufs->verifybuff +  buff_parameter -> recvbuff_disp[j] + z;
            std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
        }
        std::cout << "  ";
    }

    std::cout << dec << std::endl;
}


__global__ void identity_kernel(int32_t *sendbuff, size_t count) {
    int i0 = blockIdx.x*(count/gridDim.x);
    i0 += blockIdx.x < count%gridDim.x ? blockIdx.x : count%gridDim.x;
    int i1 = (blockIdx.x+1)*(count/gridDim.x);
    i1 += blockIdx.x+1 < count%gridDim.x ? blockIdx.x+1 : count%gridDim.x;
    int i = i0 + threadIdx.x;
    while(i < i1) {
        sendbuff[i] = sendbuff[i] * 1;
        i += blockDim.x;
    }
}


struct buffer_ptr_t init_buffers(struct buffer_parameter_t * buff_parameter, uint server_n, uint gpu_n, uint rank){
    // allocate memory
    uint data_type_size = sizeof(int32_t);
    void * sendbuff = NULL, * recvbuff = NULL, * lbsend = NULL, * lbrecv = NULL, * crosbuff1 = NULL, *crosbuff2 = NULL, * rstrbuff = NULL, * verifybuff = NULL;
    RCCLCHECK(hipMalloc((void **)&sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size));
    RCCLCHECK(hipMalloc((void **)&recvbuff, buff_parameter -> recvbuff_total_sz * data_type_size));
    if (buff_parameter -> lbsend_total_sz) RCCLCHECK(hipMalloc((void **)&lbsend, buff_parameter -> lbsend_total_sz * data_type_size));
    if (buff_parameter -> lbrecv_total_sz) RCCLCHECK(hipMalloc((void **)&lbrecv, buff_parameter -> lbrecv_total_sz * data_type_size));
    if (buff_parameter -> crosbuff_total_sz) {
        RCCLCHECK(hipMalloc((void **)&crosbuff1, buff_parameter -> crosbuff_total_sz * data_type_size));
        RCCLCHECK(hipMalloc((void **)&crosbuff2, buff_parameter -> crosbuff_total_sz * data_type_size));
    }
    if (buff_parameter -> rstrbuff_total_sz) RCCLCHECK(hipMalloc((void **)&rstrbuff, buff_parameter -> rstrbuff_total_sz * data_type_size));
    verifybuff = malloc(buff_parameter -> recvbuff_total_sz * data_type_size);
    // initialize memory
    // if(rank== 0)std::cout << "recv buff total size: " <<  buff_parameter -> recvbuff_total_sz << std::endl;
    RCCLCHECK(hipMemset(recvbuff, 0, buff_parameter -> recvbuff_total_sz * data_type_size));



    uint local_rank_id = rank % gpu_n;
    uint dim = server_n * gpu_n;
    if (buff_parameter -> lbsend_total_sz){
        int32_t * host_lbsend = new int32_t[buff_parameter -> lbsend_total_sz];
        memset(host_lbsend, 0, buff_parameter -> lbsend_total_sz * data_type_size);
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
                for (uint s = 0; s < server_n; s++){
                    uint64_t disp = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    uint64_t sz = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                    uint64_t offset = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s];
                    uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                    for (uint64_t z = 0; z < sz; z ++){
                        uint unique_data =  ((rank & 0xff) << 24) + ((dst_gpu_global_id & 0xff) << 16) + ((z + offset) & 0xffff);
                        host_lbsend[disp + z] = unique_data;
                    }
                }
            }
        }
        RCCLCHECK(hipMemcpy(lbsend, (void *) host_lbsend, buff_parameter -> lbsend_total_sz * data_type_size, hipMemcpyHostToDevice));
    }

    int32_t * host_sendbuff = new int32_t[buff_parameter -> sendbuff_total_sz];
    memset(host_sendbuff, 0, buff_parameter -> sendbuff_total_sz * data_type_size);
    for (uint i = 0; i < dim; i++){
        uint64_t disp = buff_parameter -> sendbuff_region[i].src_gpu_disp[local_rank_id];
        uint64_t sz = buff_parameter -> sendbuff_region[i].src_gpu_sz[local_rank_id];
        uint64_t offset = buff_parameter -> sendbuff_region[i].src_gpu_offset[local_rank_id];
        for (uint64_t j = 0; j < sz; j++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + ((j + offset) & 0xffff);
            host_sendbuff[disp + j] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(sendbuff, (void *) host_sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size, hipMemcpyHostToDevice));
    memset(verifybuff, 0,  buff_parameter -> recvbuff_total_sz);
    for (uint i = 0; i < dim; i++){
        uint64_t disp = buff_parameter -> recvbuff_disp[i];
        uint64_t sz = buff_parameter -> recvbuff_sz[i];
        for (uint64_t j = 0; j < sz; j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [disp + j] = unique_data;
        }
    }

    RCCLCHECK(hipDeviceSynchronize());
    delete[] host_sendbuff;

    struct buffer_ptr_t bufs = {
        .sendbuff = sendbuff,
        .recvbuff = recvbuff,
        .lbsend = lbsend,
        .lbrecv = lbrecv,
        .crosbuff1 = crosbuff1,
        .crosbuff2 = crosbuff2,
        .rstrbuff = rstrbuff,
        .verifybuff = verifybuff
    };
    return bufs;
}


struct buffer_ptr_t init_buffers_ablation(struct buffer_parameter_t * buff_parameter, uint server_n, uint gpu_n, uint rank){
    // allocate memory
    uint data_type_size = sizeof(int32_t);
    void * sendbuff = NULL, * recvbuff = NULL, * lbsend = NULL, * lbrecv = NULL, * crosbuff = NULL, * rstrbuff = NULL, * verifybuff = NULL;
    RCCLCHECK(hipMalloc((void **)&sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size));
    RCCLCHECK(hipMalloc((void **)&recvbuff, buff_parameter -> recvbuff_total_sz * data_type_size));
    if (buff_parameter -> lbsend_total_sz) RCCLCHECK(hipMalloc((void **)&lbsend, buff_parameter -> lbsend_total_sz * data_type_size));
    if (buff_parameter -> lbrecv_total_sz) RCCLCHECK(hipMalloc((void **)&lbrecv, buff_parameter -> lbrecv_total_sz * data_type_size));
    if (buff_parameter -> ablation_crosbuff_total_sz) {
        RCCLCHECK(hipMalloc((void **)&crosbuff, buff_parameter -> ablation_crosbuff_total_sz * data_type_size));
    }
    if (buff_parameter -> ablation_rstrbuff_total_sz) RCCLCHECK(hipMalloc((void **)&rstrbuff, buff_parameter -> ablation_rstrbuff_total_sz * data_type_size));
    verifybuff = malloc(buff_parameter -> recvbuff_total_sz * data_type_size);
    // initialize memory
    // if(rank== 0)std::cout << "recv buff total size: " <<  buff_parameter -> recvbuff_total_sz << std::endl;
    RCCLCHECK(hipMemset(recvbuff, 0, buff_parameter -> recvbuff_total_sz * data_type_size));

    uint local_rank_id = rank % gpu_n;
    uint dim = server_n * gpu_n;
    if (buff_parameter -> lbsend_total_sz){
        int32_t * host_lbsend = new int32_t[buff_parameter -> lbsend_total_sz];
        memset(host_lbsend, 0, buff_parameter -> lbsend_total_sz * data_type_size);
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
                for (uint s = 0; s < server_n; s++){
                    uint64_t disp = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    uint64_t sz = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                    uint64_t offset = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s];
                    uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                    for (uint64_t z = 0; z < sz; z ++){
                        uint unique_data =  ((rank & 0xff) << 24) + ((dst_gpu_global_id & 0xff) << 16) + ((z + offset) & 0xffff);
                        host_lbsend[disp + z] = unique_data;
                    }
                }
            }
        }
        RCCLCHECK(hipMemcpy(lbsend, (void *) host_lbsend, buff_parameter -> lbsend_total_sz * data_type_size, hipMemcpyHostToDevice));
    }

    int32_t * host_sendbuff = new int32_t[buff_parameter -> sendbuff_total_sz];
    memset(host_sendbuff, 0, buff_parameter -> sendbuff_total_sz * data_type_size);
    for (uint i = 0; i < dim; i++){
        uint64_t disp = buff_parameter -> sendbuff_region[i].src_gpu_disp[local_rank_id];
        uint64_t sz = buff_parameter -> sendbuff_region[i].src_gpu_sz[local_rank_id];
        uint64_t offset = buff_parameter -> sendbuff_region[i].src_gpu_offset[local_rank_id];
        for (uint64_t j = 0; j < sz; j++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + ((j + offset) & 0xffff);
            host_sendbuff[disp + j] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(sendbuff, (void *) host_sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size, hipMemcpyHostToDevice));
    memset(verifybuff, 0,  buff_parameter -> recvbuff_total_sz);
    for (uint i = 0; i < dim; i++){
        uint64_t disp = buff_parameter -> recvbuff_disp[i];
        uint64_t sz = buff_parameter -> recvbuff_sz[i];
        for (uint64_t j = 0; j < sz; j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [disp + j] = unique_data;
        }
    }

    RCCLCHECK(hipDeviceSynchronize());
    delete[] host_sendbuff;

    struct buffer_ptr_t bufs = {
        .sendbuff = sendbuff,
        .recvbuff = recvbuff,
        .lbsend = lbsend,
        .lbrecv = lbrecv,
        .crosbuff1 = crosbuff,
        .crosbuff2 = NULL,
        .rstrbuff = rstrbuff,
        .verifybuff = verifybuff
    };
    return bufs;
}


struct baseline_buffer_t{
    void * sendbuff;
    void * recvbuff;
    uint64_t * send_disp;
    uint64_t * send_sz;
    uint64_t * recv_disp;
    uint64_t * recv_sz;
};

struct baseline_buffer_t init_baseline_buffers(uint64_t* workload, uint server_n, uint gpu_n, uint rank){
    uint dim = gpu_n * server_n;
    uint local_gpu_id = rank % gpu_n;
    uint data_type_size = sizeof(int32_t);
    RCCLCHECK(hipSetDevice(rank % gpu_n));
    uint64_t * send_disp = (uint64_t *)  malloc(sizeof(uint64_t) * dim);
    uint64_t * send_sz = (uint64_t *) malloc(sizeof(uint64_t) * dim);
    uint64_t * recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * dim);
    uint64_t * recv_sz = (uint64_t *) malloc(sizeof(uint64_t) * dim);
    uint64_t sdisp = 0, rdisp = 0;
    for (uint i = 0; i < dim; i ++){
        send_disp[i] = sdisp;
        send_sz[i] = workload[rank * dim + i];
        sdisp += send_sz[i];
        recv_disp[i] = rdisp;
        recv_sz[i] = workload[i * dim + rank];
        rdisp += recv_sz[i];
    }
    void * sendbuff, * recvbuff;
    RCCLCHECK(hipMalloc((void **)&sendbuff, sdisp * data_type_size));
    RCCLCHECK(hipMalloc((void **)&recvbuff, rdisp * data_type_size));
    RCCLCHECK(hipMemset(recvbuff, 0, rdisp * data_type_size));

    int32_t * host_sendbuff = new int32_t[sdisp];
    memset(host_sendbuff, 0, sdisp * data_type_size);
    for (uint i = 0; i < dim; i++){
        for (uint64_t j = 0; j < send_sz[i]; j++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + (j & 0xffff);
            host_sendbuff[send_disp[i] + j] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(sendbuff, (void *) host_sendbuff,  sdisp * data_type_size, hipMemcpyHostToDevice));
    RCCLCHECK(hipDeviceSynchronize());
    delete[] host_sendbuff;

    struct baseline_buffer_t buf = {
        .sendbuff = sendbuff,
        .recvbuff = recvbuff,
        .send_disp = send_disp,
        .send_sz = send_sz,
        .recv_disp = recv_disp,
        .recv_sz = recv_sz
    };
    return buf;
}

void free_baseline_buffers(struct baseline_buffer_t *bufs){
    RCCLCHECK(hipFree(bufs -> sendbuff));
    RCCLCHECK(hipFree(bufs -> recvbuff));
    free(bufs -> send_disp);
    free(bufs -> send_sz);
    free(bufs -> recv_disp);
    free(bufs -> recv_sz);
}

void free_buffers( struct buffer_ptr_t * bufs){
    RCCLCHECK(hipFree(bufs -> sendbuff));
    RCCLCHECK(hipFree(bufs -> recvbuff));
    if (bufs -> lbsend) RCCLCHECK(hipFree(bufs -> lbsend));
    if (bufs -> lbrecv) RCCLCHECK(hipFree(bufs -> lbrecv));
    if (bufs -> crosbuff1) RCCLCHECK(hipFree(bufs -> crosbuff1));
    if (bufs -> crosbuff2) RCCLCHECK(hipFree(bufs -> crosbuff2));
    if (bufs -> rstrbuff) RCCLCHECK(hipFree(bufs -> rstrbuff));
    free (bufs -> verifybuff);
}


struct perf_test_ret_t{
    double algbw;
    double time;
    double loadbalance_time;
    double crossnode_time;
    double restore_time;
};

struct max_sum_ret_t{
    uint64_t max;
    uint64_t sum;
};

struct max_sum_ret_t max_sum_matrix(uint64_t * workload, uint dim){
    uint64_t max = 0, sum = 0;
    for (uint i = 0; i < dim; i ++){
        for (uint j = 0; j < dim; j++){
            max = MAX(max, workload[i * dim + j]);
            sum += workload[i * dim + j];
        }
    }
    struct max_sum_ret_t r = {.max = max, .sum = sum};
    return r;
}

ncclResult_t
fastAllToAll_ablation(void* sendbuff, void* recvbuff, void * lbsend, void * lbrecv, void * crosbuff, void * rstrbuff,
    struct scheduling_result_gpu_t * gpu_sched,
    ncclDataType_t data_type, ncclComm_t comm, hipStream_t stream, hipEvent_t* ts, uint * ts_id){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = gpu_sched->rankid,
        local_rank_id = gpu_sched->rankid % gpu_sched->gpu_n,
        server_id = gpu_sched->rankid / gpu_sched->gpu_n,
        server_n = gpu_sched->server_n,
        gpu_n = gpu_sched->gpu_n,
        rankid = gpu_sched->rankid;

    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }

    // load balance
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> balance_send_n; i++){
        NCCLCHECK(ncclSend((char *)lbsend + gpu_sched -> balance_send[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_send[i].sz,
                        data_type,
                        gpu_sched -> balance_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched -> balance_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)lbrecv + gpu_sched -> balance_recv[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_recv[i].sz,
                        data_type,
                        gpu_sched -> balance_recv[i].gpu,
                        comm,
                        stream));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> balance_memcpy_n; i ++){
        RCCLCHECK(hipMemcpyWithStream((char*)sendbuff + gpu_sched -> balance_memcpy[i].dst_disp * sizeof(int32_t),
                         (char*) lbrecv + gpu_sched -> balance_memcpy[i].src_disp * sizeof(int32_t),
                          gpu_sched -> balance_memcpy[i].sz * sizeof(int32_t),
                          hipMemcpyDeviceToDevice,
                          stream));
    }

    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }

    // cross-node send
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> ablation_crossnode_send_n; i++){
        NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> ablation_crossnode_send[i].disp * sizeof(int32_t),
                        gpu_sched -> ablation_crossnode_send[i].sz,
                        data_type,
                        gpu_sched -> ablation_crossnode_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched -> ablation_crossnode_recv_n; i++){
    NCCLCHECK(ncclRecv((char *)crosbuff + gpu_sched -> ablation_crossnode_recv[i].disp * sizeof(int32_t),
                    gpu_sched -> ablation_crossnode_recv[i].sz,
                    data_type,
                    gpu_sched -> ablation_crossnode_recv[i].gpu,
                    comm,
                    stream));
    }

    NCCLCHECK(ncclGroupEnd());
    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }

    // intrinsic alltoall
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> intrinsic_send_n; i ++){
            NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> intrinsic_send[i].disp * sizeof(int32_t),
                        gpu_sched -> intrinsic_send[i].sz,
                        data_type,
                        gpu_sched -> intrinsic_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched -> intrinsic_recv_n; i++){
            NCCLCHECK(ncclRecv((char *)recvbuff + gpu_sched -> intrinsic_recv[i].disp * sizeof(int32_t),
                gpu_sched -> intrinsic_recv[i].sz,
                data_type,
                gpu_sched -> intrinsic_recv[i].gpu,
                comm,
                stream));
    }
    NCCLCHECK(ncclGroupEnd());

    // final restore
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> ablation_restore_send_n; i++){
        NCCLCHECK(ncclSend((char *)crosbuff + gpu_sched -> ablation_restore_send[i].disp * sizeof(int32_t),
                gpu_sched -> ablation_restore_send[i].sz,
                data_type,
                gpu_sched -> ablation_restore_send[i].gpu,
                comm,
                stream
        ));
    }

    for (uint i = 0; i < gpu_sched -> ablation_restore_recv_n; i++){
        NCCLCHECK(ncclRecv((char*)rstrbuff + gpu_sched -> ablation_restore_recv[i].disp * sizeof(int32_t),
                gpu_sched -> ablation_restore_recv[i].sz,
                data_type,
                gpu_sched -> ablation_restore_recv[i].gpu,
                comm,
                stream
        ));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> ablation_direct_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char *)recvbuff + gpu_sched -> ablation_direct_memcpy[i].dst_disp * sizeof(int32_t),
                (char *) crosbuff + gpu_sched -> ablation_direct_memcpy[i].src_disp * sizeof(int32_t),
                gpu_sched -> ablation_direct_memcpy[i].sz * sizeof(int32_t),
                hipMemcpyDeviceToDevice,
                stream));
    }

    for (uint i = 0; i < gpu_sched -> ablation_restore_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char *)recvbuff + gpu_sched -> ablation_restore_memcpy[i].dst_disp * sizeof(int32_t),
                (char *) rstrbuff + gpu_sched -> ablation_restore_memcpy[i].src_disp * sizeof(int32_t),
                gpu_sched -> ablation_restore_memcpy[i].sz * sizeof(int32_t),
                hipMemcpyDeviceToDevice,
                stream));
    }
    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }

    return ncclSuccess;
}



bool verify_correctness_fastalltoall_ablation(struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched,  struct buffer_parameter_t * buff_parameter, ncclComm_t comm, hipStream_t stream){
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(fastAllToAll_ablation(
        bufs->sendbuff,
        bufs->recvbuff,
        bufs->lbsend,
        bufs->lbrecv,
        bufs->crosbuff1,
        bufs->rstrbuff,
        gpu_sched,
        ncclInt32,
        comm,
        stream,
        NULL,
        NULL));

    RCCLCHECK(hipDeviceSynchronize());
    if (gpu_sched -> rankid == 0) std::cout << "transfer completed: now verify correctness..." <<std::endl;
    for (uint i = 0; i < gpu_sched -> server_n * gpu_sched -> gpu_n; i++){
        char* recvbuff = (char*)bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        char* verifybuff =  (char*) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        uint64_t sz = buff_parameter -> recvbuff_sz[i];
        if(0 != memcmp(recvbuff, verifybuff,  sz * sizeof(int32_t))){
            std::cout << "RANK " << gpu_sched -> rankid << " from rank " << i << " buffer error" << std::endl;
            std::cout<< "recvbuff:" <<std::endl;
            for (uint64_t z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
            std::cout<< "verifybuff:" <<std::endl;
            for (uint64_t z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}


struct perf_test_ret_t perf_fastalltoall_ablation(uint warmup_iters, uint perf_iters, struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched, ncclComm_t comm, hipStream_t stream, uint64_t buff_size){

    hipEvent_t start_event, end_event;
    hipEvent_t *tss = new hipEvent_t[perf_iters * 4];
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    for (uint i = 0; i < perf_iters * 4; i++){
        RCCLCHECK(hipEventCreate(&tss[i]));
    }
    uint ts_id = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(fastAllToAll_ablation(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff1,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm,
            stream,
            NULL,
            NULL));

    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(fastAllToAll_ablation(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff1,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm,
            stream,
            tss,
            &ts_id));

    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / (gpu_sched->server_n * gpu_sched->gpu_n);

    float et[3] = {0.0, 0.0, 0.0}, st[3] = {0.0, 0.0, 0.0};
    for (uint i = 0; i < perf_iters; i++){
        for (uint j = 0; j < 3; j++){
            hipEventElapsedTime(&et[j], tss[i * 4 + j], tss[i * 4 + j + 1]);
            st[j] += et[j];
        }
    }

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    for (uint i = 0; i < perf_iters * 4; i++){
        RCCLCHECK(hipEventDestroy(tss[i]));
    }
    delete[] tss;
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = st[0] / perf_iters, .crossnode_time = st[1] / perf_iters, .restore_time = st[2] / perf_iters};
    return r;
}



void test_fastalltoall_ablation(ncclComm_t comm,  hipStream_t stream, uint nranks, uint rank){
    uint64_t * workload = new uint64_t[nranks * nranks];
    if (rank == 0) uniform_distribution(workload, nranks, pow(2,18));
    MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
    uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
    if (rank == 0) print_matrix(workload, nranks, nranks);

    // scheduler is deterministic given a workload
    uint server_n = 2, gpu_n = 8;
    struct GlobalScheduler scheduler;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
    run_scheduler_ablation(&scheduler);
    scheduler.sched->rankid = rank;

    RCCLCHECK(hipSetDevice(rank % gpu_n));
    struct buffer_ptr_t buffer_ptrs = init_buffers_ablation(scheduler.buff_parameter, server_n, gpu_n, rank);

    // verify correctness
    if (rank == 0) std::cout << "TESTING CORRECTNESS" << std::endl;
    int correctness_this_rank = verify_correctness_fastalltoall_ablation(&buffer_ptrs, scheduler.gpu_sched, scheduler.buff_parameter, comm, stream);
    int * correctness;
    if (rank == 0) correctness = new int[nranks];
    MPI_Gather(&correctness_this_rank, 1, MPI_INT, correctness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int correctness_n = 0;
        for (uint i = 0; i < nranks; i++) {
            correctness_n += correctness[i];
            if ( correctness[i] == 0) std::cout << "rank " << i << " fails" << std::endl;
        }
        std::cout << "fastalltoall ablation CORRECTNESS: " << ((correctness_n == nranks) ? "succeed" : "fail") << std::endl;
        delete[] correctness;
    }

    // perf test
    struct perf_test_ret_t r = perf_fastalltoall_ablation(25, 25, &buffer_ptrs, scheduler.gpu_sched, comm, stream, buff_size);
    if(rank == 0) std::cout << "fastalltoall ablation algbw: " << r.algbw << " GBps" <<std::endl
                    << "total time: " << r.time << " ms, lb: " << r.loadbalance_time << " ms, cross: "
                    << r.crossnode_time << " ms, restore: " << r.restore_time << " ms"<< std::endl;

    delete[] workload;
    free_buffers(&buffer_ptrs);
    free_global_scheduler(&scheduler);
}


ncclResult_t
fastAllToAll(void* sendbuff, void* recvbuff, void * lbsend, void * lbrecv, void * crosbuff1, void * crosbuff2, void * rstrbuff,
    struct scheduling_result_gpu_t * gpu_sched,
    ncclDataType_t data_type, ncclComm_t comm, hipStream_t stream, hipEvent_t* ts, uint * ts_id){


    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = gpu_sched->rankid,
        local_rank_id = gpu_sched->rankid % gpu_sched->gpu_n,
        erver_id = gpu_sched->rankid / gpu_sched->gpu_n,
        server_n = gpu_sched->server_n,
        gpu_n = gpu_sched->gpu_n,
        rankid = gpu_sched->rankid;

    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }

    // load balance
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> balance_send_n; i++){
        NCCLCHECK(ncclSend((char *)lbsend + gpu_sched -> balance_send[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_send[i].sz,
                        data_type,
                        gpu_sched -> balance_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched -> balance_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)lbrecv + gpu_sched -> balance_recv[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_recv[i].sz,
                        data_type,
                        gpu_sched -> balance_recv[i].gpu,
                        comm,
                        stream));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> balance_memcpy_n; i ++){
        RCCLCHECK(hipMemcpyWithStream((char*)sendbuff + gpu_sched -> balance_memcpy[i].dst_disp * sizeof(int32_t),
                         (char*) lbrecv + gpu_sched -> balance_memcpy[i].src_disp * sizeof(int32_t),
                          gpu_sched -> balance_memcpy[i].sz * sizeof(int32_t),
                          hipMemcpyDeviceToDevice,
                          stream));
    }

    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id)++;
    }

    // intrinsic alltoall and first cross node
    struct scheduling_step_gpu_t * cur_step = &gpu_sched -> steps[0];
    uint64_t cross_send_sz = cur_step -> crossnode_send.sz;
    uint64_t cross_recv_sz = cur_step -> crossnode_recv.sz;
    void * cur_crosbuff = crosbuff1, * prev_crosbuff = crosbuff1;
    NCCLCHECK(ncclGroupStart());
    // first cross-node send
    // std::cout << "[cross node] " << " send sz: " << cross_send_sz << " recv sz: " << cross_recv_sz << std::endl;
    NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                    cross_send_sz,
                    data_type,
                    cur_step -> crossnode_send.gpu,
                    comm,
                    stream));

    NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
            cross_recv_sz,
            data_type,
            cur_step -> crossnode_recv.gpu,
            comm,
            stream));
    // intrinsic alltoall
    uint m = 0, n = 0;
    while (m + n < gpu_sched -> intrinsic_send_n + gpu_sched ->intrinsic_recv_n){
        if (m < gpu_sched -> intrinsic_send_n){
            // std:: cout << "rank: "<< global_rank_id << " , dst gpu: " <<  gpu_sched -> intrinsic_send[m].gpu << ", disp: " << gpu_sched -> intrinsic_send[m].disp << ", sz: " << gpu_sched -> intrinsic_send[m].sz std::endl;
            NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> intrinsic_send[m].disp * sizeof(int32_t),
                        gpu_sched -> intrinsic_send[m].sz,
                        data_type,
                        gpu_sched -> intrinsic_send[m].gpu,
                        comm,
                        stream));
            m ++;
        }
        if ( n < gpu_sched ->intrinsic_recv_n){
            NCCLCHECK(ncclRecv((char *)recvbuff + gpu_sched -> intrinsic_recv[n].disp * sizeof(int32_t),
                        gpu_sched -> intrinsic_recv[n].sz,
                        data_type,
                        gpu_sched -> intrinsic_recv[n].gpu,
                        comm,
                        stream));
            n ++;
        }
    }

    NCCLCHECK(ncclGroupEnd());
    // middle steps

    for (uint step_id = 1; step_id < gpu_sched -> step_n - 1; step_id ++){
        // std::cout << "step id: " << step_id  << std::endl;
        cur_step = &gpu_sched -> steps[step_id];
        cross_send_sz = cur_step -> crossnode_send.sz;
        cross_recv_sz = cur_step -> crossnode_recv.sz;
        cur_crosbuff = (step_id % 2 == 1) ? crosbuff2 : crosbuff1;
        prev_crosbuff = (step_id % 2 == 1) ? crosbuff1 : crosbuff2;
        NCCLCHECK(ncclGroupStart());
        // cross node transfer
        if (cross_send_sz > 0){
            NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                            cross_send_sz,
                            data_type,
                            cur_step -> crossnode_send.gpu,
                            comm,
                            stream));
        }
        if (cross_recv_sz > 0){
            NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
                    cross_recv_sz,
                    data_type,
                    cur_step -> crossnode_recv.gpu,
                    comm,
                    stream));
        }

        // NCCLCHECK(ncclGroupEnd());

        // NCCLCHECK(ncclGroupStart());

        // data restore of previous step
        for (uint i = 0; i < cur_step -> restore_send_n; i ++){
            NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[i].disp * sizeof(int32_t),
                cur_step -> restore_send[i].sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm,
                stream));
        }

        for (uint i = 0; i < cur_step -> restore_recv_n; i++){
            NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * sizeof(int32_t),
                cur_step -> restore_recv[i].sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm,
                stream));
        }
        NCCLCHECK(ncclGroupEnd());

        for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
            RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice));
        }

        for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
            RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                        (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                        cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                        hipMemcpyDeviceToDevice));
        }
    }

    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }


    // final data restore
    prev_crosbuff = ((gpu_sched -> step_n - 1) % 2 == 1) ? crosbuff1 : crosbuff2;
    cur_step = &gpu_sched -> steps[gpu_sched -> step_n - 1];
    NCCLCHECK(ncclGroupStart());
    m = 0;
    n = 0;
    while (m + n < cur_step -> restore_send_n + cur_step -> restore_recv_n){
        if (m < cur_step -> restore_send_n){
            NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[m].disp * sizeof(int32_t),
                cur_step -> restore_send[m].sz,
                data_type,
                cur_step -> restore_send[m].gpu,
                comm,
                stream));
            m ++;
        }

        if ( n <  cur_step -> restore_recv_n){
            NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[n].disp * sizeof(int32_t),
            cur_step -> restore_recv[n].sz,
            data_type,
            cur_step -> restore_recv[n].gpu,
            comm,
            stream));
            n ++;
        }
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                hipMemcpyDeviceToDevice,
                stream));
    }
    for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice,
                    stream));
    }
    if (ts) {
        hipEventRecord(ts[*ts_id], stream);
        (*ts_id) ++;
    }
    return ncclSuccess;
}


ncclResult_t
fastAllToAll2(void* sendbuff, void* recvbuff, void * lbsend, void * lbrecv, void * crosbuff1, void * crosbuff2, void * rstrbuff,
    struct scheduling_result_gpu_t * gpu_sched,
    ncclDataType_t data_type, ncclComm_t comm, ncclComm_t comm2, hipStream_t stream1, hipStream_t stream2, hipEvent_t* ts, uint * ts_id){


    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = gpu_sched->rankid,
        local_rank_id = gpu_sched->rankid % gpu_sched->gpu_n,
        erver_id = gpu_sched->rankid / gpu_sched->gpu_n,
        server_n = gpu_sched->server_n,
        gpu_n = gpu_sched->gpu_n,
        rankid = gpu_sched->rankid;

    if (ts) {
        hipEventRecord(ts[*ts_id], stream1);
        (*ts_id) ++;
    }

    // load balance
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> balance_send_n; i++){
        NCCLCHECK(ncclSend((char *)lbsend + gpu_sched -> balance_send[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_send[i].sz,
                        data_type,
                        gpu_sched -> balance_send[i].gpu,
                        comm,
                        stream1));
    }

    for (uint i = 0; i < gpu_sched -> balance_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)lbrecv + gpu_sched -> balance_recv[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_recv[i].sz,
                        data_type,
                        gpu_sched -> balance_recv[i].gpu,
                        comm,
                        stream1));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> balance_memcpy_n; i ++){
        RCCLCHECK(hipMemcpyWithStream((char*)sendbuff + gpu_sched -> balance_memcpy[i].dst_disp * sizeof(int32_t),
                         (char*) lbrecv + gpu_sched -> balance_memcpy[i].src_disp * sizeof(int32_t),
                          gpu_sched -> balance_memcpy[i].sz * sizeof(int32_t),
                          hipMemcpyDeviceToDevice,
                          stream1));
    }
    RCCLCHECK(hipStreamSynchronize(stream1));

    if (ts) {
        hipEventRecord(ts[*ts_id], stream1);
        (*ts_id)++;
    }

    // intrinsic alltoall and first cross node
    struct scheduling_step_gpu_t * cur_step = &gpu_sched -> steps[0];
    uint64_t cross_send_sz = cur_step -> crossnode_send.sz;
    uint64_t cross_recv_sz = cur_step -> crossnode_recv.sz;
    void * cur_crosbuff = crosbuff1, * prev_crosbuff = crosbuff1;
    NCCLCHECK(ncclGroupStart());
    // first cross-node send
    // std::cout << "[cross node] " << " send sz: " << cross_send_sz << " recv sz: " << cross_recv_sz << std::endl;
    NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                    cross_send_sz,
                    data_type,
                    cur_step -> crossnode_send.gpu,
                    comm2,
                    stream2));

    NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
            cross_recv_sz,
            data_type,
            cur_step -> crossnode_recv.gpu,
            comm2,
            stream2));
    // intrinsic alltoall
    for (uint i = 0; i < gpu_sched -> intrinsic_send_n; i++){
        NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> intrinsic_send[i].disp * sizeof(int32_t),
                    gpu_sched -> intrinsic_send[i].sz,
                    data_type,
                    gpu_sched -> intrinsic_send[i].gpu,
                    comm,
                    stream1));
    }


    for (uint i = 0; i < gpu_sched -> intrinsic_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)recvbuff + gpu_sched -> intrinsic_recv[i].disp * sizeof(int32_t),
            gpu_sched -> intrinsic_recv[i].sz,
            data_type,
            gpu_sched -> intrinsic_recv[i].gpu,
            comm,
            stream1));
    }
    NCCLCHECK(ncclGroupEnd());
    RCCLCHECK(hipStreamSynchronize(stream2));
    // middle steps

    for (uint step_id = 1; step_id < gpu_sched -> step_n - 1; step_id ++){
        // std::cout << "step id: " << step_id  << std::endl;
        cur_step = &gpu_sched -> steps[step_id];
        cross_send_sz = cur_step -> crossnode_send.sz;
        cross_recv_sz = cur_step -> crossnode_recv.sz;
        cur_crosbuff = (step_id % 2 == 1) ? crosbuff2 : crosbuff1;
        prev_crosbuff = (step_id % 2 == 1) ? crosbuff1 : crosbuff2;
        NCCLCHECK(ncclGroupStart());
        // cross node transfer
        if (cross_send_sz > 0){
            NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                            cross_send_sz,
                            data_type,
                            cur_step -> crossnode_send.gpu,
                            comm2,
                            stream2));
        }
        if (cross_recv_sz > 0){
            NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
                    cross_recv_sz,
                    data_type,
                    cur_step -> crossnode_recv.gpu,
                    comm2,
                    stream2));
        }

        // data restore of previous step
        for (uint i = 0; i < cur_step -> restore_send_n; i ++){
            NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[i].disp * sizeof(int32_t),
                cur_step -> restore_send[i].sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm,
                stream1));
        }

        for (uint i = 0; i < cur_step -> restore_recv_n; i++){
            NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * sizeof(int32_t),
                cur_step -> restore_recv[i].sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm,
                stream1));
        }
        NCCLCHECK(ncclGroupEnd());

        for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
            RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice,
                    stream1));
        }

        for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
            RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                        (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                        cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                        hipMemcpyDeviceToDevice,
                        stream1));
        }
        RCCLCHECK(hipStreamSynchronize(stream2));
        RCCLCHECK(hipStreamSynchronize(stream1));
    }

    if (ts) {
        hipEventRecord(ts[*ts_id], stream1);
        (*ts_id) ++;
    }


    // final data restore
    prev_crosbuff = ((gpu_sched -> step_n - 1) % 2 == 1) ? crosbuff1 : crosbuff2;
    cur_step = &gpu_sched -> steps[gpu_sched -> step_n - 1];
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < cur_step -> restore_send_n; i ++){
         NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[i].disp * sizeof(int32_t),
                cur_step -> restore_send[i].sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm,
                stream1));

    }
    for (uint i = 0; i < cur_step -> restore_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * sizeof(int32_t),
                cur_step -> restore_recv[i].sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm,
                stream1));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                hipMemcpyDeviceToDevice,
                stream1));
    }
    for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice,
                    stream1));
    }
    if (ts) {
        hipEventRecord(ts[*ts_id], stream1);
        (*ts_id) ++;
    }
    return ncclSuccess;
}

bool verify_correctness_fastalltoall(struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched,  struct buffer_parameter_t * buff_parameter, ncclComm_t comm1, ncclComm_t comm2, hipStream_t stream1, hipStream_t stream2){
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(fastAllToAll2(
        bufs->sendbuff,
        bufs->recvbuff,
        bufs->lbsend,
        bufs->lbrecv,
        bufs->crosbuff1,
        bufs->crosbuff2,
        bufs->rstrbuff,
        gpu_sched,
        ncclInt32,
        comm1,
        comm2,
        stream1,
        stream2,
        NULL,
        NULL));

    RCCLCHECK(hipDeviceSynchronize());
    if (gpu_sched -> rankid == 0) std::cout << "transfer completed: now verify correctness..." <<std::endl;
    for (uint i = 0; i < gpu_sched -> server_n * gpu_sched -> gpu_n; i++){
        char* recvbuff = (char*)bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        char* verifybuff =  (char*) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] * sizeof(int32_t);
        uint64_t sz = buff_parameter -> recvbuff_sz[i];
        if(0 != memcmp(recvbuff, verifybuff,  sz * sizeof(int32_t))){
            std::cout << "RANK " << gpu_sched -> rankid << " from rank " << i << " buffer error" << std::endl;
            std::cout<< "recvbuff:" <<std::endl;
            for (uint64_t z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->recvbuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
            std::cout<< "verifybuff:" <<std::endl;
            for (uint64_t z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->verifybuff +  buff_parameter -> recvbuff_disp[i] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}


struct perf_test_ret_t perf_fastalltoall(uint warmup_iters, uint perf_iters, struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched, ncclComm_t comm1, ncclComm_t comm2, hipStream_t stream1, hipStream_t stream2, uint64_t buff_size){

    hipEvent_t start_event, end_event;
    hipEvent_t * tss = new hipEvent_t[perf_iters * 4];
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    for (uint i = 0; i < perf_iters * 4; i++){
        RCCLCHECK(hipEventCreate(&tss[i]));
    }
    uint ts_id = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(fastAllToAll2(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff1,
            bufs->crosbuff2,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm1,
            comm2,
            stream1,
            stream2,
            NULL,
            NULL));

    }
    RCCLCHECK(hipEventRecord(start_event, stream1));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(fastAllToAll2(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff1,
            bufs->crosbuff2,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm1,
            comm2,
            stream1,
            stream2,
            tss,
            &ts_id));

    }
    RCCLCHECK(hipEventRecord(end_event, stream1));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / (gpu_sched->server_n * gpu_sched->gpu_n);

    float et[3] = {0.0, 0.0, 0.0}, st[3] = {0.0, 0.0, 0.0};
    for (uint i = 0; i < perf_iters; i++){
        for (uint j = 0; j < 3; j++){
            hipEventElapsedTime(&et[j], tss[i * 4 + j], tss[i * 4 + j + 1]);
            st[j] += et[j];
        }
    }

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    for (uint i = 0; i < perf_iters * 4; i++){
        RCCLCHECK(hipEventDestroy(tss[i]));
    }
    delete[] tss;
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = st[0] / perf_iters, .crossnode_time = st[1] / perf_iters, .restore_time = st[2] / perf_iters};
    return r;
}


void test_fastalltoall(ncclComm_t comm1, ncclComm_t comm2,  hipStream_t stream1, hipStream_t stream2, uint nranks, uint rank){
    uint64_t * workload = new uint64_t[nranks * nranks];
    if (rank == 0) uniform_distribution(workload, nranks, pow(2,22));
    MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
    uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
    if (rank == 0) print_matrix(workload, nranks, nranks);

    // scheduler is deterministic given a workload
    uint server_n = 2, gpu_n = 8;
    struct GlobalScheduler scheduler;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
    run_scheduler(&scheduler);
    scheduler.sched->rankid = rank;

    RCCLCHECK(hipSetDevice(rank % gpu_n));
    struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);

    // verify correctness
    if (rank == 0) std::cout << "TESTING CORRECTNESS" << std::endl;
    int correctness_this_rank = verify_correctness_fastalltoall(&buffer_ptrs, scheduler.gpu_sched, scheduler.buff_parameter, comm1, comm2, stream1, stream2);
    int * correctness;
    if (rank == 0) correctness = new int[nranks];
    MPI_Gather(&correctness_this_rank, 1, MPI_INT, correctness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int correctness_n = 0;
        for (uint i = 0; i < nranks; i++) {
            correctness_n += correctness[i];
            if ( correctness[i] == 0) std::cout << "rank " << i << " fails" << std::endl;
        }
        std::cout << "fastalltoall ablation CORRECTNESS: " << ((correctness_n == nranks) ? "succeed" : "fail") << std::endl;
        delete[] correctness;
    }

    // perf test
    struct perf_test_ret_t r = perf_fastalltoall(25, 25, &buffer_ptrs, scheduler.gpu_sched, comm1, comm2, stream1, stream2, buff_size);
    if(rank == 0) std::cout << "fastalltoall ablation algbw: " << r.algbw << " GBps" <<std::endl
                    << "total time: " << r.time << " ms, lb: " << r.loadbalance_time << " ms, cross: "
                    << r.crossnode_time << " ms, restore: " << r.restore_time << " ms"<< std::endl;

    delete[] workload;
    free_buffers(&buffer_ptrs);
    free_global_scheduler(&scheduler);
}

ncclResult_t
fanoutAllToAll(struct baseline_buffer_t * param, uint dim, uint rank, uint gpu_n,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    // uint server_id = rank / gpu_n;
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < dim; i++){
        // uint local_gpu = server_id * gpu_n + i;
        NCCLCHECK(ncclSend(
            (char *)param->sendbuff + param->send_disp[i] * sizeof(int32_t),
            param->send_sz[i],
            datatype,
            i,
            comm,
            stream
        ));
        NCCLCHECK(ncclRecv(
            (char *)param->recvbuff + param->recv_disp[i] * sizeof(int32_t),
            param->recv_sz[i],
            datatype,
            i,
            comm,
            stream
        ));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}


struct perf_test_ret_t perf_fanout(uint warmup_iters, uint perf_iters, struct baseline_buffer_t * param, ncclComm_t comm, hipStream_t stream, uint64_t buff_size, uint dim, uint rank, uint gpu_n){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(fanoutAllToAll(
            param,
            dim,
            rank,
            gpu_n,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(fanoutAllToAll(
            param,
            dim,
            rank,
            gpu_n,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / dim;

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = 0.0, .crossnode_time = avg_time, .restore_time = 0.0};
    return r;
}



ncclResult_t
spreadoutAllToAll(struct baseline_buffer_t * param, uint dim, uint rank, uint gpu_n,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    // uint server_id = rank / gpu_n;
    for (uint i = 1; i < dim; i++){
        // uint local_gpu = server_id * gpu_n + i;
        uint dst_gpu = (rank + i) % dim, src_gpu = (rank + dim - i) % dim;
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(
            (char *)param->sendbuff + param->send_disp[dst_gpu] * sizeof(int32_t),
            param->send_sz[dst_gpu],
            datatype,
            dst_gpu,
            comm,
            stream
        ));
        NCCLCHECK(ncclRecv(
            (char *)param->recvbuff + param->recv_disp[src_gpu] * sizeof(int32_t),
            param->recv_sz[src_gpu],
            datatype,
            src_gpu,
            comm,
            stream
        ));
        NCCLCHECK(ncclGroupEnd());
    }

    return ncclSuccess;
}



struct perf_test_ret_t perf_spreadout(uint warmup_iters, uint perf_iters, struct baseline_buffer_t * param, ncclComm_t comm, hipStream_t stream, uint64_t buff_size, uint dim, uint rank, uint gpu_n){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(spreadoutAllToAll(
            param,
            dim,
            rank,
            gpu_n,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(spreadoutAllToAll(
            param,
            dim,
            rank,
            gpu_n,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / dim;

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = 0, .crossnode_time = avg_time, .restore_time = 0.0};
    return r;
}


struct incast_buffer_t{
    // for incast sender
    void * sendbuff;
    // for incast receiver
    void * recvbuff;
    uint count;
};


struct incast_buffer_t init_incast_buffers(uint send_sz, uint rank, uint gpu_n, uint incast_receiver, uint incast_degree){

    uint data_type_size = sizeof(int32_t);
    RCCLCHECK(hipSetDevice(rank % gpu_n));
    if (rank / gpu_n == 0){
        if (rank == incast_receiver){
            void * recvbuff;
            RCCLCHECK(hipMalloc((void **)&recvbuff, send_sz * incast_degree * data_type_size));
            RCCLCHECK(hipMemset(recvbuff, 0, send_sz * incast_degree * data_type_size));
            RCCLCHECK(hipDeviceSynchronize());
            struct incast_buffer_t r = {
                .sendbuff = NULL,
                .recvbuff = recvbuff,
                .count = send_sz
            };
            return r;
        }
    }else{
        uint send_rank = (rank / gpu_n - 1) * gpu_n + rank % gpu_n;
        if (send_rank < incast_degree){
            void * sendbuff;
            RCCLCHECK(hipMalloc((void **)&sendbuff, send_sz * data_type_size));
            int32_t * host_sendbuff = new int32_t[send_sz];
            memset(host_sendbuff, 0, send_sz * data_type_size);
            for (uint j = 0; j < send_sz; j++){
                int32_t unique_data = ((rank & 0xff) << 24) + ((incast_receiver & 0xff) << 16) + (j & 0xffff);
                host_sendbuff[j] = unique_data;
            }
            RCCLCHECK(hipMemcpy(sendbuff, (void *) host_sendbuff,  send_sz * data_type_size, hipMemcpyHostToDevice));
            RCCLCHECK(hipDeviceSynchronize());
            struct incast_buffer_t r = {
                .sendbuff = sendbuff,
                .recvbuff = NULL,
                .count = send_sz
            };
            return r;
        }
    }

    struct incast_buffer_t r = {
        .sendbuff = NULL,
        .recvbuff = NULL,
        .count = 0
    };
    return r;
}


void  free_incast_buffers(struct incast_buffer_t * buf){
    if (buf -> count > 0){
        if (buf -> sendbuff) RCCLCHECK(hipFree(buf -> sendbuff));
        if (buf -> recvbuff) RCCLCHECK(hipFree(buf -> recvbuff));
    }
}


ncclResult_t
incast(struct incast_buffer_t * param, uint rank, uint gpu_n, uint incast_receiver, uint incast_degree,
        ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    // incast receiver can be any one from rank 0 - 7
    // incast sender are ranks from second and third node
    if (incast_degree < 1 || incast_degree > 16){
        std::cout << "INVALID INCAST DEGREE: " << incast_degree << std::endl;
        return ncclSuccess;
    }
    if (rank / gpu_n == 0){
        if (rank == incast_receiver){
            NCCLCHECK(ncclGroupStart());
            for (uint i = 0; i < incast_degree; i++){
                uint src_gpu = (i / gpu_n + 1) * gpu_n + (i % gpu_n);
                NCCLCHECK(ncclRecv(
                    (char *)param->recvbuff + i * param->count * sizeof(int32_t),
                    param->count,
                    datatype,
                    src_gpu,
                    comm,
                    stream
                ));
            }
            NCCLCHECK(ncclGroupEnd());
        }
    }else{
        uint send_rank = (rank / gpu_n - 1) * gpu_n + rank % gpu_n;
        if (send_rank < incast_degree){
            std::cout << "rank : " << rank << " send!" <<std::endl;
            // send data to the receiving incast receiver
            NCCLCHECK(ncclSend(
                param->sendbuff,
                param->count,
                datatype,
                incast_receiver,
                comm,
                stream
            ));
        }
    }
    return ncclSuccess;
}


struct perf_test_ret_t perf_incast(uint warmup_iters, uint perf_iters, struct incast_buffer_t * param, ncclComm_t comm, hipStream_t stream, uint rank, uint gpu_n, uint incast_receiver, uint incast_degree){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(incast(
            param,
            rank,
            gpu_n,
            incast_receiver,
            incast_degree,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(incast(
            param,
            rank,
            gpu_n,
            incast_receiver,
            incast_degree,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double)  param -> count * (1e-6) * incast_degree * sizeof(int32_t)/ avg_time;

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
    return r;
}


void perftest_incast(ncclComm_t comm,  hipStream_t stream, uint nranks, uint rank){
    uint test_times = 200;
    uint warmup_iters = 10, perf_iters = 10;
    uint incast_receiver = 0;
    uint gpu_n = 8;
    for (uint incast_degree = 2; incast_degree <= 16; incast_degree += 2){
        struct incast_buffer_t buffers = init_incast_buffers(pow(2, 24), rank, gpu_n, incast_receiver, incast_degree);
        struct perf_test_ret_t ret = perf_incast(warmup_iters, perf_iters, &buffers, comm, stream, rank, gpu_n, incast_receiver, incast_degree);
        if (rank == incast_receiver){
            std:: cout << "incast degree: " << incast_degree << " , algbw: " << ret.algbw << " GBps" << std::endl;
        }
        free_incast_buffers(&buffers);
    }
}

struct perf_result_t{
    uint64_t sz; // in Bytes
    double skewness;
    struct perf_test_ret_t fastalltoall;
    struct perf_test_ret_t fastalltoall_ablation;
    struct perf_test_ret_t fanout;
    struct perf_test_ret_t spreadout;
};


// void perftest_uniform_speedup_cdf(ncclComm_t comm,  hipStream_t stream, uint nranks, uint rank){
//     uint test_times = 20;
//     uint warmup_iters = 20, perf_iters = 20;
//     uint64_t * workload = new uint64_t[nranks * nranks];
//     uint64_t avg_buff_sz = 0;
//     uint server_n = 3, gpu_n = 8;
//     std::vector<struct speed_up_per_workload> speedups;
//     if (rank == 0) std::cout << "PERFORMANCE TEST RANDOM CDF: total " << test_times << " workloads" << std::endl;
//     for (uint power = 15; power <= 27; power++){
//         for (uint i = 0; i < test_times; i++){
//             // 1000 different workload
//             if (rank == 0) uniform_distribution(workload, nranks, pow(2, power));
//             MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
//             struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
//             uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
//             struct GlobalScheduler scheduler;
//             init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
//             run_scheduler(&scheduler);
//             scheduler.sched->rankid = rank;
//             RCCLCHECK(hipSetDevice(rank % gpu_n));
//             struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
//             struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
//             struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, stream, buff_size);
//             struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct speed_up_per_workload spdup = {
//                 .sz = buff_size,
//                 .skewness = 0.0,
//                 .fastalltoall_algbw = fastalltoall.algbw,
//                 .fanout_algbw = fanout.algbw,
//                 .spreadout_algbw = spreadout.algbw,
//                 .fanout_spdup = fastalltoall.algbw / fanout.algbw,
//                 .spreadout_spdup = fastalltoall.algbw / spreadout.algbw
//             };
//             if (rank == 0) std::cout << "workload " << i << " sz: " << pow(2, power) << ": fanout speedup: " << spdup.fanout_spdup << " , spreadout speedup: " << spdup.spreadout_spdup << std::endl;
//             speedups.push_back(spdup);
//             free_buffers(&buffer_ptrs);
//             free_baseline_buffers(&baseline_buffer_ptrs);
//             free_global_scheduler(&scheduler);
//         }
//     }
//     delete[] workload;

//     if (rank == 0){
//         ofstream cdf_txt;
//         cdf_txt.open ("../fastalltoall_result/uniform_speedup_cdf.txt");
//         for (uint i = 0; i < speedups.size(); i ++){
//             cdf_txt << speedups[i].sz << " " << speedups[i].fastalltoall_algbw << " " << speedups[i].fanout_algbw << " " << speedups[i].spreadout_algbw << " " <<speedups[i].fanout_spdup << " " <<  speedups[i].spreadout_spdup << std::endl;
//         }
//         cdf_txt.close();
//     }
// }



// void perftest_zipf_speedup_cdf(ncclComm_t comm,  hipStream_t stream, uint nranks, uint rank){
//     uint test_times = 100;
//     uint warmup_iters = 10, perf_iters = 10;
//     uint64_t * workload = new uint64_t[nranks * nranks];
//     uint64_t avg_buff_sz = 0;
//     uint server_n = 3, gpu_n = 8;
//     double skewnesses[20] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};
//     std::vector<struct speed_up_per_workload> speedups;
//     if (rank == 0) std::cout << "PERFORMANCE TEST ZIPF CDF: total " << test_times << " workloads" << std::endl;
//     for (uint j = 0; j < 11; j ++){
//         double skewness = skewnesses[j];
//         for (uint i = 0; i < test_times; i++){
//             // 1000 different workload
//             if (rank == 0) zipf_distribution(workload, skewness, nranks, 33554432);
//             MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
//             struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
//             uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
//             struct GlobalScheduler scheduler;
//             init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
//             run_scheduler(&scheduler);
//             scheduler.sched->rankid = rank;
//             RCCLCHECK(hipSetDevice(rank % gpu_n));
//             struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
//             struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
//             struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, stream, buff_size);
//             struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct speed_up_per_workload spdup = {
//                 .sz = buff_size,
//                 .skewness = skewness,
//                 .fastalltoall_algbw = fastalltoall.algbw,
//                 .fanout_algbw = fanout.algbw,
//                 .spreadout_algbw = spreadout.algbw,
//                 .fanout_spdup = fastalltoall.algbw / fanout.algbw,
//                 .spreadout_spdup = fastalltoall.algbw / spreadout.algbw
//             };
//             if (rank == 0) std::cout << "workload " << i << ", skewness: " << skewness <<": fanout speedup: " << spdup.fanout_spdup << " , spreadout speedup: " << spdup.spreadout_spdup << std::endl;
//             speedups.push_back(spdup);
//             free_buffers(&buffer_ptrs);
//             free_baseline_buffers(&baseline_buffer_ptrs);
//             free_global_scheduler(&scheduler);
//         }

//     }

//     delete[] workload;

//     if (rank == 0){
//         ofstream cdf_txt;
//         cdf_txt.open ("../fastalltoall_result/skewness_speedup_cdf.txt");
//         for (uint i = 0; i < speedups.size(); i ++){
//             cdf_txt << speedups[i].sz << " " << speedups[i].skewness << " " << speedups[i].fastalltoall_algbw << " " << speedups[i].fanout_algbw << " " << speedups[i].spreadout_algbw << " " <<speedups[i].fanout_spdup << " " <<  speedups[i].spreadout_spdup << std::endl;
//         }
//         cdf_txt.close();
//     }
// }



// void perftest_fixed_speedup_cdf(ncclComm_t comm,  hipStream_t stream, uint nranks, uint rank){
//     uint test_times = 100;
//     uint warmup_iters = 25, perf_iters = 25;
//     uint64_t * workload = new uint64_t[nranks * nranks];
//     uint64_t avg_buff_sz = 0;
//     uint server_n = 3, gpu_n = 8;
//     std::vector<struct speed_up_per_workload> speedups;
//     if (rank == 0) std::cout << "PERFORMANCE TEST FIXED CDF: total " << test_times << " workloads" << std::endl;
//     uint power_list[] = {20, 21, 24};
//     for (uint p = 0; p < 3; p++){
//         uint power = power_list[p];
//         for (uint i = 0; i < test_times; i++){
//             // 1000 different workload
//             if (rank == 0) fixed_distribution(workload, nranks, pow(2, power));
//             MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
//             struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
//             uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
//             struct GlobalScheduler scheduler;
//             init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
//             run_scheduler(&scheduler);
//             scheduler.sched->rankid = rank;
//             RCCLCHECK(hipSetDevice(rank % gpu_n));
//             struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
//             struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
//             struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, stream, buff_size);
//             struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
//             struct speed_up_per_workload spdup = {
//                 .sz = buff_size,
//                 .skewness = 0.0,
//                 .fastalltoall_algbw = fastalltoall.algbw,
//                 .fanout_algbw = fanout.algbw,
//                 .spreadout_algbw = spreadout.algbw,
//                 .fanout_spdup = fastalltoall.algbw / fanout.algbw,
//                 .spreadout_spdup = fastalltoall.algbw / spreadout.algbw
//             };
//             if (rank == 0) std::cout << "workload " << i << " , size: " << pow(2, power) << ", fanout speedup: " << spdup.fanout_spdup << " , spreadout speedup: " << spdup.spreadout_spdup << std::endl;
//             speedups.push_back(spdup);
//             free_buffers(&buffer_ptrs);
//             free_baseline_buffers(&baseline_buffer_ptrs);
//             free_global_scheduler(&scheduler);
//         }
//     }

//     delete[] workload;

//     if (rank == 0){
//         ofstream cdf_txt;
//         cdf_txt.open ("../fastalltoall_result/fixed_speedup_cdf.txt");
//         for (uint i = 0; i < speedups.size(); i ++){
//             cdf_txt << speedups[i].sz << " " << speedups[i].fastalltoall_algbw << " " << speedups[i].fanout_algbw << " " << speedups[i].spreadout_algbw << " " <<speedups[i].fanout_spdup << " " <<  speedups[i].spreadout_spdup << std::endl;
//         }
//         cdf_txt.close();
//     }
// }


void perftest_fixed_transfer_sz(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (transfer sz): fixed workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    for (uint power = 16; power <= 26; power++){
        for (uint i = 0; i < test_times; i++){
            if (rank == 0) fixed_distribution(workload, nranks, pow(2, power));
            MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
            uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
            struct GlobalScheduler scheduler, scheduler2;
            init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
            init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
            scheduler.sched->rankid = rank;
            scheduler2.sched->rankid = rank;
            run_scheduler(&scheduler);
            run_scheduler_ablation(&scheduler2);
            RCCLCHECK(hipSetDevice(rank % gpu_n));

            struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
            free_buffers(&buffer_ptrs);
            struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
            free_buffers(&buffer_ptrs_ablation);
            struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
            struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            free_baseline_buffers(&baseline_buffer_ptrs);

            struct perf_result_t cur_r = {
                .sz = buff_size,
                .skewness = 0.0,
                .fastalltoall = fastalltoall,
                .fastalltoall_ablation = fastalltoall_ablation,
                .fanout = fanout,
                .spreadout = spreadout,
            };
            if (rank == 0) {
                char prog[20];
                sprintf(prog, "%u/%u", i+1, test_times);
                std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                  << setfill(' ') << setw(20) << "-"
                  << setfill(' ') << setw(20) << prog
                  << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                  << setfill(' ') << setw(20) << cur_r.fanout.algbw
                  << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                  << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
            }
            speedups.push_back(cur_r);
            free_global_scheduler(&scheduler);
            free_global_scheduler(&scheduler2);
        }
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_fixed_sz_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}


void perftest_random_transfer_sz(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (transfer sz): random workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    for (uint power = 16; power <= 26; power++){
        for (uint i = 0; i < test_times; i++){
            if (rank == 0) uniform_distribution2(workload, nranks, pow(2, power));
            MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
            uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
            struct GlobalScheduler scheduler, scheduler2;
            init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
            init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
            scheduler.sched->rankid = rank;
            scheduler2.sched->rankid = rank;
            run_scheduler(&scheduler);
            run_scheduler_ablation(&scheduler2);
            RCCLCHECK(hipSetDevice(rank % gpu_n));

            struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
            free_buffers(&buffer_ptrs);
            struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
            free_buffers(&buffer_ptrs_ablation);
            struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
            struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            free_baseline_buffers(&baseline_buffer_ptrs);

            struct perf_result_t cur_r = {
                .sz = buff_size,
                .skewness = 0.0,
                .fastalltoall = fastalltoall,
                .fastalltoall_ablation = fastalltoall_ablation,
                .fanout = fanout,
                .spreadout = spreadout,
            };
            if (rank == 0) {
                char prog[20];
                sprintf(prog, "%u/%u", i+1, test_times);
                std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                  << setfill(' ') << setw(20) << "R"
                  << setfill(' ') << setw(20) << prog
                  << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                  << setfill(' ') << setw(20) << cur_r.fanout.algbw
                  << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                  << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
            }
            speedups.push_back(cur_r);
            free_global_scheduler(&scheduler);
            free_global_scheduler(&scheduler2);
        }
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_random_sz_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}

void perftest_skewed_transfer_sz(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, double skewness, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (transfer sz): skewed workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    for (uint power = 16; power <= 26; power++){
        for (uint i = 0; i < test_times; i++){
            if (rank == 0) zipf_distribution2(workload, skewness, nranks, pow(2, power));
            MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
            uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
            struct GlobalScheduler scheduler, scheduler2;
            init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
            init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
            scheduler.sched->rankid = rank;
            scheduler2.sched->rankid = rank;
            run_scheduler(&scheduler);
            run_scheduler_ablation(&scheduler2);
            RCCLCHECK(hipSetDevice(rank % gpu_n));

            struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
            free_buffers(&buffer_ptrs);
            struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
            free_buffers(&buffer_ptrs_ablation);
            struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
            struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            free_baseline_buffers(&baseline_buffer_ptrs);

            struct perf_result_t cur_r = {
                .sz = buff_size,
                .skewness = skewness,
                .fastalltoall = fastalltoall,
                .fastalltoall_ablation = fastalltoall_ablation,
                .fanout = fanout,
                .spreadout = spreadout,
            };
            if (rank == 0) {
                char prog[20];
                sprintf(prog, "%u/%u", i+1, test_times);
                std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                  << setfill(' ') << setw(20) << skewness
                  << setfill(' ') << setw(20) << prog
                  << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                  << setfill(' ') << setw(20) << cur_r.fanout.algbw
                  << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                  << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
            }
            speedups.push_back(cur_r);
            free_global_scheduler(&scheduler);
            free_global_scheduler(&scheduler2);
        }
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_skewed_sz__%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}


void perftest_skewness(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (skewness): skewed workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    double skewnesses[20] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};
    uint64_t per_gpu_sz = pow(2, 22);

    for (uint j = 0; j < 11; j ++){
        double skewness = skewnesses[j];
        for (uint i = 0; i < test_times; i++){
            if (rank == 0) zipf_distribution2(workload, skewness, nranks, per_gpu_sz);
            MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
            uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
            struct GlobalScheduler scheduler, scheduler2;
            init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
            init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
            scheduler.sched->rankid = rank;
            scheduler2.sched->rankid = rank;
            run_scheduler(&scheduler);
            run_scheduler_ablation(&scheduler2);
            RCCLCHECK(hipSetDevice(rank % gpu_n));

            struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
            free_buffers(&buffer_ptrs);
            struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
            struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
            free_buffers(&buffer_ptrs_ablation);
            struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
            struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            free_baseline_buffers(&baseline_buffer_ptrs);

            struct perf_result_t cur_r = {
                .sz = buff_size,
                .skewness = skewness,
                .fastalltoall = fastalltoall,
                .fastalltoall_ablation = fastalltoall_ablation,
                .fanout = fanout,
                .spreadout = spreadout,
            };
            if (rank == 0) {
                char prog[20];
                sprintf(prog, "%u/%u", i+1, test_times);
                std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                  << setfill(' ') << setw(20) << skewness
                  << setfill(' ') << setw(20) << prog
                  << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                  << setfill(' ') << setw(20) << cur_r.fanout.algbw
                  << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                  << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
            }
            speedups.push_back(cur_r);
            free_global_scheduler(&scheduler);
            free_global_scheduler(&scheduler2);
        }
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_skewness_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}


void perftest_random_cdf(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (cdf): random workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    uint64_t per_gpu_sz = pow(2, 22);

    for (uint i = 0; i < test_times; i++){
        if (rank == 0) uniform_distribution2(workload, nranks, per_gpu_sz);
        MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
        uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
        struct GlobalScheduler scheduler, scheduler2;
        init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
        init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
        scheduler.sched->rankid = rank;
        scheduler2.sched->rankid = rank;
        run_scheduler(&scheduler);
        run_scheduler_ablation(&scheduler2);
        RCCLCHECK(hipSetDevice(rank % gpu_n));

        struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
        free_buffers(&buffer_ptrs);
        struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
        free_buffers(&buffer_ptrs_ablation);
        struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
        struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
        struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
        free_baseline_buffers(&baseline_buffer_ptrs);

        struct perf_result_t cur_r = {
            .sz = buff_size,
            .skewness = 0.0,
            .fastalltoall = fastalltoall,
            .fastalltoall_ablation = fastalltoall_ablation,
            .fanout = fanout,
            .spreadout = spreadout,
        };
        if (rank == 0) {
            char prog[20];
            sprintf(prog, "%u/%u", i+1, test_times);
            std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                << setfill(' ') << setw(20) << "R"
                << setfill(' ') << setw(20) << prog
                << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                << setfill(' ') << setw(20) << cur_r.fanout.algbw
                << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
        }
        speedups.push_back(cur_r);
        free_global_scheduler(&scheduler);
        free_global_scheduler(&scheduler2);
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_random_cdf_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}



void perftest_skewed_cdf(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 10, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct perf_result_t> speedups;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "PERFORMANCE TEST (cdf): skewed workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Skewness"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "FLASH (GBps)"
                  << setfill(' ') << setw(20) << "Fanout (GBps)"
                  << setfill(' ') << setw(20) << "FLASH-A (GBps)"
                  << setfill(' ') << setw(20) << "Spreadout (GBbps)" << std::endl;
    }
    uint64_t per_gpu_sz = pow(2, 22);
    double skewness = 0.8;

    for (uint i = 0; i < test_times; i++){
        if (rank == 0) zipf_distribution2(workload, skewness, nranks, per_gpu_sz);
        MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
        uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
        struct GlobalScheduler scheduler, scheduler2;
        init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
        init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
        scheduler.sched->rankid = rank;
        scheduler2.sched->rankid = rank;
        run_scheduler(&scheduler);
        run_scheduler_ablation(&scheduler2);
        RCCLCHECK(hipSetDevice(rank % gpu_n));

        struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, comm2, stream, stream2, buff_size);
        free_buffers(&buffer_ptrs);
        struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream, buff_size);
        free_buffers(&buffer_ptrs_ablation);
        struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
        struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
        struct perf_test_ret_t spreadout = perf_spreadout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
        free_baseline_buffers(&baseline_buffer_ptrs);

        struct perf_result_t cur_r = {
            .sz = buff_size,
            .skewness = skewness,
            .fastalltoall = fastalltoall,
            .fastalltoall_ablation = fastalltoall_ablation,
            .fanout = fanout,
            .spreadout = spreadout,
        };
        if (rank == 0) {
            char prog[20];
            sprintf(prog, "%u/%u", i+1, test_times);
            std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                << setfill(' ') << setw(20) << skewness
                << setfill(' ') << setw(20) << prog
                << setfill(' ') << setw(20) << cur_r.fastalltoall.algbw
                << setfill(' ') << setw(20) << cur_r.fanout.algbw
                << setfill(' ') << setw(20) << cur_r.fastalltoall_ablation.algbw
                << setfill(' ') << setw(20) << cur_r.spreadout.algbw  << std::endl;
        }
        speedups.push_back(cur_r);
        free_global_scheduler(&scheduler);
        free_global_scheduler(&scheduler2);
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_skewed_cdf_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < speedups.size(); i ++){
            cdf_txt << speedups[i].sz << " "
                    << speedups[i].skewness << " "
                    << speedups[i].fastalltoall.algbw << " "
                    << speedups[i].fanout.algbw << " "
                    << speedups[i].fastalltoall_ablation.algbw << " "
                    << speedups[i].spreadout.algbw << " "
                    << speedups[i].fastalltoall.time << " "
                    << speedups[i].fastalltoall.loadbalance_time << " "
                    << speedups[i].fastalltoall.crossnode_time << " "
                    << speedups[i].fastalltoall.restore_time << " "
                    << speedups[i].fanout.time << " "
                    << speedups[i].fastalltoall_ablation.time << " "
                    << speedups[i].fastalltoall_ablation.loadbalance_time << " "
                    << speedups[i].fastalltoall_ablation.crossnode_time << " "
                    << speedups[i].fastalltoall_ablation.restore_time << " "
                    << speedups[i].spreadout.time << std::endl;
        }
        cdf_txt.close();
    }
}


void ablation_test(ncclComm_t comm, ncclComm_t comm2, hipStream_t stream1, hipStream_t stream2, uint nranks, uint rank){
    uint test_times = 100;
    uint warmup_iters = 25, perf_iters = 25;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    uint server_n = 2, gpu_n = 8;

    if (rank == 0) std::cout << "ABLATION TEST: total " << test_times << " workloads" << std::endl;
    // fixed workload
    for (uint i = 0; i < test_times; i++){
        if (rank == 0) fixed_distribution(workload, nranks, pow(2, 22));
        MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
        uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
        struct GlobalScheduler scheduler1, scheduler2;
        init_global_scheduler(&scheduler1, server_n, gpu_n, workload, rank);
        init_global_scheduler(&scheduler2, server_n, gpu_n, workload, rank);
        scheduler1.sched->rankid = rank;
        scheduler2.sched->rankid = rank;
        run_scheduler(&scheduler1);
        run_scheduler_ablation(&scheduler2);
        RCCLCHECK(hipSetDevice(rank % gpu_n));
        struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler1.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall = perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler1.gpu_sched, comm, comm2, stream1, stream2, buff_size);
        free_buffers(&buffer_ptrs);
        struct buffer_ptr_t buffer_ptrs_ablation = init_buffers_ablation(scheduler2.buff_parameter, server_n, gpu_n, rank);
        struct perf_test_ret_t fastalltoall_ablation = perf_fastalltoall_ablation(warmup_iters, perf_iters, &buffer_ptrs_ablation, scheduler2.gpu_sched, comm, stream1, buff_size);
        free_buffers(&buffer_ptrs_ablation);
        struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
        struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream1, buff_size, server_n * gpu_n, rank, gpu_n);
        free_baseline_buffers(&baseline_buffer_ptrs);

        if(rank == 0) {
            std::cout << setfill(' ') << setw(20) << "algbw (GBps)" << setfill(' ') << setw(20) << "time (ms)" <<setfill(' ') << setw(20) << "lb (ms)" <<setfill(' ') << setw(20)<< "crossnode (ms)"<<setfill(' ') << setw(20) << "restore (ms)" << std::endl
            << setfill(' ') << setw(20) <<scheduler1.opt.algbw_limit << setfill(' ') << setw(20) <<scheduler1.opt.crossnode_time_limit << std::endl
            << setfill(' ') << setw(20) <<fastalltoall.algbw << setfill(' ') << setw(20) <<fastalltoall.time << setfill(' ') << setw(20) <<fastalltoall.loadbalance_time << setfill(' ') << setw(20) <<fastalltoall.crossnode_time << setfill(' ') << setw(20) <<fastalltoall.restore_time << std::endl
            << setfill(' ') << setw(20) <<fastalltoall_ablation.algbw << setfill(' ') << setw(20) <<fastalltoall_ablation.time << setfill(' ') << setw(20) <<fastalltoall_ablation.loadbalance_time << setfill(' ') << setw(20) <<fastalltoall_ablation.crossnode_time << setfill(' ') << setw(20) <<fastalltoall_ablation.restore_time << std::endl
            << setfill(' ') << setw(20) <<fanout.algbw << setfill(' ') << setw(20) << fanout.time << setfill(' ') << setw(20) <<fanout.loadbalance_time << setfill(' ') << setw(20) <<fanout.crossnode_time << setfill(' ') << setw(20) <<fanout.restore_time << std::endl;
        }
        free_global_scheduler(&scheduler1);
        free_global_scheduler(&scheduler2);
    }
    delete[] workload;
}

struct overhead_per_workload{
    uint64_t sz;
    uint rank;
    uint node;
    long sched;   // us
    uint64_t fastalltoall_mem;
    uint64_t baseline_mem;
    double mem_ratio;
};


struct incast_bw_loss_ret{
    uint64_t sz;
    double algbw;
    uint nack_n;
};


uint read_nack_counter(uint local_rank){
    vector<string> rnics = {"rocep102s0f0", "rocep134s0f0", "rocep163s0f0", "rocep195s0f0", "rocep230s0f0", "rocep35s0", "rocep67s0f1", "rocep6s0f1"};
    string rnic_counter = "/sys/class/infiniband/" + rnics[local_rank] + "/ports/1/hw_counters/packet_seq_err";
    ifstream sys_ctnr_f (rnic_counter);
    string val;
    getline(sys_ctnr_f, val);
    sys_ctnr_f.close();
    return atoi(val.c_str());
}

void incast_bw_loss_test(ncclComm_t comm, ncclComm_t comm2,  hipStream_t stream, hipStream_t stream2, uint nranks, uint rank, uint server_n, uint gpu_n, uint test_times = 10){
    uint warmup_iters = 0, perf_iters = 10;
    uint64_t * workload = new uint64_t[nranks * nranks];
    uint64_t avg_buff_sz = 0;
    std::vector<struct incast_bw_loss_ret> ret;
    if (rank == 0) {
        std::cout <<  "------------------------------------------------------------" << std::endl <<
                        "NACK and Transfer sz relation: fixed workload - total " << test_times << " samples" << std::endl <<
                       "------------------------------------------------------------" << std::endl;
        std::cout << setfill(' ') << setw(20) << "GPU Size (MB)"
                  << setfill(' ') << setw(20) << "Progress"
                  << setfill(' ') << setw(20) << "NACK#"
                  << setfill(' ') << setw(20) << "Fanout (GBps)" << std::endl;
    }

    uint nack_ctnr1 = 0, nack_ctnr2 = 0, nack_ctnr_sum = 0;
    for (uint power = 16; power <= 27; power++){
        for (uint i = 0; i < test_times; i++){
            if (rank == 0) fixed_distribution(workload, nranks, pow(2, power));
            MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
            uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
            RCCLCHECK(hipSetDevice(rank % gpu_n));


            struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, server_n, gpu_n, rank);
            nack_ctnr1 = read_nack_counter(rank % gpu_n);
            struct perf_test_ret_t fanout = perf_fanout(warmup_iters, perf_iters, &baseline_buffer_ptrs, comm, stream, buff_size, server_n * gpu_n, rank, gpu_n);
            nack_ctnr2 = read_nack_counter(rank % gpu_n);
            nack_ctnr2 -= nack_ctnr1;
            free_baseline_buffers(&baseline_buffer_ptrs);

            uint* ctnr ;
            if (rank == 0) ctnr = new uint[nranks];
            MPI_Gather(&nack_ctnr2, sizeof(uint), MPI_BYTE, ctnr, sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
            nack_ctnr_sum = 0;
            if (rank == 0){
                for (uint i = 0; i < nranks; i++){
                    nack_ctnr_sum += ctnr[i];
                }
                delete[] ctnr;
            }

            struct incast_bw_loss_ret cur_r = {
                .sz = buff_size,
                .algbw = fanout.algbw,
                .nack_n = nack_ctnr_sum
            };
            if (rank == 0) {
                char prog[20];
                sprintf(prog, "%u/%u", i+1, test_times);
                std::cout << setfill(' ') << setw(20) << buff_size / 1e6 / (server_n * gpu_n)
                  << setfill(' ') << setw(20) << prog
                  << setfill(' ') << setw(20) << cur_r.nack_n
                  << setfill(' ') << setw(20) << cur_r.algbw << std::endl;
            }
            ret.push_back(cur_r);
        }
    }

    delete[] workload;

    if (rank == 0){
        const auto p1 = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        char filename[200];
        sprintf(filename, "../fastalltoall_result/%un_nack_%ld.txt", server_n, ts);
        ofstream cdf_txt;
        cdf_txt.open (filename);
        for (uint i = 0; i < ret.size(); i ++){
            cdf_txt << ret[i].sz << " "
                    << ret[i].algbw << " "
                    << ret[i].nack_n << std::endl;
        }
        cdf_txt.close();
    }
}

void mem_sched_overhead(uint rank){
    uint test_times = 100;
    uint gpu_n = 8;
    std::vector<struct overhead_per_workload> overheads;

    for (uint node = 3; node <= 8; node ++){
        uint nranks = node * gpu_n;
        uint64_t * workload = new uint64_t[nranks * nranks];
        for (uint power = 15; power <= 24; power++){
            for (uint i = 0; i < test_times; i ++){
                if (rank == 0) uniform_distribution(workload, nranks, pow(2, power));
                MPI_Bcast(workload, nranks * nranks * sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
                uint64_t buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes

                struct GlobalScheduler scheduler;
                init_global_scheduler(&scheduler, node, gpu_n, workload, rank);
                auto t1 = high_resolution_clock::now();
                run_scheduler(&scheduler);
                auto t2 = high_resolution_clock::now();
                scheduler.sched->rankid = rank;
                // initialize buffers to make sure GPU has enough memory
                RCCLCHECK(hipSetDevice(rank % gpu_n));
                struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, node, gpu_n, rank);
                struct baseline_buffer_t baseline_buffer_ptrs = init_baseline_buffers(workload, node, gpu_n, rank);
                free_buffers(&buffer_ptrs);
                free_baseline_buffers(&baseline_buffer_ptrs);
                free_global_scheduler(&scheduler);

                uint64_t fastalltoall_mem = ((uint64_t)scheduler.buff_parameter->sendbuff_total_sz +
                                             (uint64_t)scheduler.buff_parameter->recvbuff_total_sz +
                                             (uint64_t)scheduler.buff_parameter->lbsend_total_sz +
                                             (uint64_t)scheduler.buff_parameter->crosbuff_total_sz * 2 +
                                             (uint64_t)scheduler.buff_parameter->rstrbuff_total_sz) * sizeof(int32_t);
                uint64_t baseline_mem = ((uint64_t)baseline_buffer_ptrs.send_disp[nranks - 1] +
                                        (uint64_t)baseline_buffer_ptrs.send_sz[nranks - 1] +
                                        (uint64_t) baseline_buffer_ptrs.recv_disp[nranks - 1] +
                                        (uint64_t) baseline_buffer_ptrs.recv_sz[nranks - 1]) * sizeof(int32_t);


                struct overhead_per_workload * o;
                if (rank == 0) o = new overhead_per_workload[24];   // only have 24 gpus in the testbed
                struct overhead_per_workload o_this_rank = {
                    .sz = buff_size,
                    .rank = rank,
                    .node = node,
                    .sched = duration_cast<microseconds>(t2 - t1).count(),
                    .fastalltoall_mem = fastalltoall_mem,
                    .baseline_mem = baseline_mem,
                    .mem_ratio = (double) fastalltoall_mem / (double)baseline_mem
                };
                MPI_Gather(&o_this_rank, sizeof(struct overhead_per_workload), MPI_BYTE, o, sizeof(struct overhead_per_workload), MPI_BYTE, 0, MPI_COMM_WORLD);
                if (rank == 0){
                    for (uint j = 0; j < 24; j++){
                        overheads.push_back(o[j]);
                    }
                    delete[] o;
                }

                if (rank == 0) std::cout << "workload " << i << " , size: " << pow(2, power) << ", node: " << node << " , scheduler runtime: " <<  duration_cast<microseconds>(t2 - t1).count() << " us, mem ratio: " <<  (double) fastalltoall_mem / (double)baseline_mem << std::endl;
            }
        }
        delete[] workload;
    }

    if (rank == 0){
        ofstream overhead_txt;
        overhead_txt.open ("../fastalltoall_result/overhead.txt");
        for (uint i = 0; i < overheads.size(); i ++){
            overhead_txt << overheads[i].sz << " " << overheads[i].rank << " " << overheads[i].node << " " << overheads[i].sched << " " <<overheads[i].fastalltoall_mem << " " <<  overheads[i].baseline_mem << " " << overheads[i].mem_ratio << std::endl;
        }
        overhead_txt.close();
    }
}



int main(int argc, char* argv[]) {
    srand((unsigned)time(0));
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Print off a hello world message
    int dev_n = 0;
    RCCLCHECK(hipGetDeviceCount(&dev_n));
    RCCLCHECK(hipSetDevice(rank % dev_n));
    std::cout << "Rank " << rank << " out of " << nranks << " successfully set device" << std::endl;

    // Initialize Communicator
    ncclComm_t comm;
    ncclUniqueId ncclId;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&ncclId));
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCLCHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));
    ncclComm_t comm2;
    ncclCommSplit(comm, 0, rank, &comm2, NULL);


    // hipStream_t stream;
    // RCCLCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipStream_t stream1, stream2;
    RCCLCHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
    RCCLCHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));

    uint server_n = 4, gpu_n = 8;

    // TEST PROGRAMs
    perftest_fixed_transfer_sz(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 10);
    perftest_random_transfer_sz(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 10);
    perftest_skewed_transfer_sz(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 0.8, 10);
    perftest_skewness(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 10);
    perftest_random_cdf(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 10);
    perftest_skewed_cdf(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 10);

    // incast_bw_loss_test(comm, comm2, stream1, stream2, nranks, rank, server_n, gpu_n, 4);
    // perftest_uniform_speedup_cdf(comm, stream, nranks, rank);
    // perftest_fixed_speedup_cdf(comm, stream, nranks, rank);
    // perftest_zipf_speedup_cdf(comm, stream, nranks, rank);
    // mem_sched_overhead(rank);
    // perftest_incast(comm, stream, nranks, rank);
    // test_fastalltoall(comm, comm2, stream1, stream2, nranks, rank);
    // test_fastalltoall_ablation(comm, stream1, nranks, rank);
    // ablation_test(comm, comm2, stream1, stream2, nranks, rank);


    NCCLCHECK(ncclCommFinalize(comm2));
    NCCLCHECK(ncclCommDestroy(comm2));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    // RCCLCHECK(hipStreamDestroy(stream));
    RCCLCHECK(hipStreamDestroy(stream1));
    RCCLCHECK(hipStreamDestroy(stream2));

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}