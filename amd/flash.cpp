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
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>




namespace py = pybind11;
using namespace torch::indexing;

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


inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
#if defined(RCCL_FLOAT8)
  case ncclFp8E4M3:
  case ncclFp8E5M2:
#endif
    return 1;
  case ncclFloat16:
#if defined(RCCL_BFLOAT16)
  case ncclBfloat16:
#endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

ncclResult_t
fastAllToAll(
    void* sendbuff,
    void* recvbuff,
    void * lbsend,
    void * lbrecv,
    void * crosbuff1,
    void * crosbuff2,
    void * rstrbuff,
    struct scheduling_result_gpu_t * gpu_sched,
    int h_sz,
    ncclDataType_t data_type,
    ncclComm_t comm[2],
    hipStream_t stream1,
    hipStream_t stream2){


    uint global_rank_id = gpu_sched->rankid,
        local_rank_id = gpu_sched->rankid % gpu_sched->gpu_n,
        erver_id = gpu_sched->rankid / gpu_sched->gpu_n,
        server_n = gpu_sched->server_n,
        gpu_n = gpu_sched->gpu_n,
        rankid = gpu_sched->rankid;

    // load balance
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> balance_send_n; i++){
        NCCLCHECK(ncclSend((char *)lbsend + gpu_sched -> balance_send[i].disp * h_sz * ncclTypeSize(data_type),
                        gpu_sched -> balance_send[i].sz * h_sz,
                        data_type,
                        gpu_sched -> balance_send[i].gpu,
                        comm[0],
                        stream1));
    }

    for (uint i = 0; i < gpu_sched -> balance_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)lbrecv + gpu_sched -> balance_recv[i].disp * h_sz * ncclTypeSize(data_type),
                        gpu_sched -> balance_recv[i].sz * h_sz,
                        data_type,
                        gpu_sched -> balance_recv[i].gpu,
                        comm[0],
                        stream1));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> balance_memcpy_n; i ++){
        RCCLCHECK(hipMemcpyWithStream((char*)sendbuff + gpu_sched -> balance_memcpy[i].dst_disp * h_sz * ncclTypeSize(data_type),
                         (char*) lbrecv + gpu_sched -> balance_memcpy[i].src_disp * h_sz * ncclTypeSize(data_type),
                          gpu_sched -> balance_memcpy[i].sz * h_sz * ncclTypeSize(data_type),
                          hipMemcpyDeviceToDevice,
                          stream1));
    }
    RCCLCHECK(hipStreamSynchronize(stream1));

    // intrinsic alltoall and first cross node
    struct scheduling_step_gpu_t * cur_step = &gpu_sched -> steps[0];
    uint64_t cross_send_sz = cur_step -> crossnode_send.sz;
    uint64_t cross_recv_sz = cur_step -> crossnode_recv.sz;
    void * cur_crosbuff = crosbuff1, * prev_crosbuff = crosbuff1;
    NCCLCHECK(ncclGroupStart());
    // first cross-node send
    // std::cout << "[cross node] " << " send sz: " << cross_send_sz << " recv sz: " << cross_recv_sz << std::endl;
    NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * h_sz * ncclTypeSize(data_type),
                    cross_send_sz * h_sz,
                    data_type,
                    cur_step -> crossnode_send.gpu,
                    comm[0],
                    stream1));

    NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * h_sz * ncclTypeSize(data_type),
            cross_recv_sz * h_sz,
            data_type,
            cur_step -> crossnode_recv.gpu,
            comm[0],
            stream1));
    // intrinsic alltoall
    for (uint i = 0; i < gpu_sched -> intrinsic_send_n; i++){
        NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> intrinsic_send[i].disp * h_sz * ncclTypeSize(data_type),
                    gpu_sched -> intrinsic_send[i].sz * h_sz,
                    data_type,
                    gpu_sched -> intrinsic_send[i].gpu,
                    comm[1],
                    stream2));
    }

    for (uint i = 0; i < gpu_sched -> intrinsic_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)recvbuff + gpu_sched -> intrinsic_recv[i].disp * h_sz * ncclTypeSize(data_type),
            gpu_sched -> intrinsic_recv[i].sz * h_sz,
            data_type,
            gpu_sched -> intrinsic_recv[i].gpu,
            comm[1],
            stream2));
    }
    NCCLCHECK(ncclGroupEnd());
    RCCLCHECK(hipStreamSynchronize(stream1));
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
            NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * h_sz * ncclTypeSize(data_type),
                            cross_send_sz * h_sz,
                            data_type,
                            cur_step -> crossnode_send.gpu,
                            comm[0],
                            stream1));
        }
        if (cross_recv_sz > 0){
            NCCLCHECK(ncclRecv((char *)cur_crosbuff + cur_step -> crossnode_recv.disp * h_sz * ncclTypeSize(data_type),
                    cross_recv_sz * h_sz,
                    data_type,
                    cur_step -> crossnode_recv.gpu,
                    comm[0],
                    stream1));
        }

        // data restore of previous step
        for (uint i = 0; i < cur_step -> restore_send_n; i ++){
            NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[i].disp * h_sz * ncclTypeSize(data_type),
                cur_step -> restore_send[i].sz * h_sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm[1],
                stream2));
        }

        for (uint i = 0; i < cur_step -> restore_recv_n; i++){
            NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * h_sz * ncclTypeSize(data_type),
                cur_step -> restore_recv[i].sz * h_sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm[1],
                stream2));
        }
        NCCLCHECK(ncclGroupEnd());

        for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
            RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * h_sz * ncclTypeSize(data_type),
                    (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * h_sz * ncclTypeSize(data_type),
                    cur_step -> direct_memcpy[i].sz * h_sz * ncclTypeSize(data_type),
                    hipMemcpyDeviceToDevice,
                    stream2));
        }

        for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
            RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * h_sz * ncclTypeSize(data_type),
                        (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * h_sz * ncclTypeSize(data_type),
                        cur_step -> restore_memcpy[i].sz * h_sz * ncclTypeSize(data_type),
                        hipMemcpyDeviceToDevice,
                        stream2));
        }
        RCCLCHECK(hipStreamSynchronize(stream1));
        RCCLCHECK(hipStreamSynchronize(stream2));
    }

    // final data restore
    prev_crosbuff = ((gpu_sched -> step_n - 1) % 2 == 1) ? crosbuff1 : crosbuff2;
    cur_step = &gpu_sched -> steps[gpu_sched -> step_n - 1];
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < cur_step -> restore_send_n; i ++){
         NCCLCHECK(ncclSend((char *)prev_crosbuff + cur_step -> restore_send[i].disp * h_sz * ncclTypeSize(data_type),
                cur_step -> restore_send[i].sz * h_sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm[0],
                stream1));

    }
    for (uint i = 0; i < cur_step -> restore_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * h_sz * ncclTypeSize(data_type),
                cur_step -> restore_recv[i].sz * h_sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm[0],
                stream1));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * h_sz * ncclTypeSize(data_type),
                (char*) prev_crosbuff + cur_step -> direct_memcpy[i].src_disp * h_sz * ncclTypeSize(data_type),
                cur_step -> direct_memcpy[i].sz * h_sz * ncclTypeSize(data_type),
                hipMemcpyDeviceToDevice,
                stream1));
    }
    for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
        RCCLCHECK(hipMemcpyWithStream((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * h_sz * ncclTypeSize(data_type),
                    (char*) rstrbuff + cur_step -> restore_memcpy[i].src_disp * h_sz * ncclTypeSize(data_type),
                    cur_step -> restore_memcpy[i].sz * h_sz * ncclTypeSize(data_type),
                    hipMemcpyDeviceToDevice,
                    stream1));
    }
    RCCLCHECK(hipStreamSynchronize(stream1));
    return ncclSuccess;
}


struct flash_buffer_ptr_t{
    void * sendbuff;
    void * recvbuff;
    void * lbsend;
    void * lbrecv;
    void * crosbuff1;
    void * crosbuff2;
    void * rstrbuff;
};


class flash_t{
private:
    ncclComm_t comm[2];
    hipStream_t stream[2];
    int rank;
    int nrank;
    int server_n;
    int gpu_n;
    uint64_t * workload;
    struct GlobalScheduler * scheduler;
    struct flash_buffer_ptr_t * buf_ptrs;
    int hidden_sz;
    ncclDataType_t dtype;
public:
    flash_t(int this_rank, int total_rank, int s_n, int g_n, int h_sz, int d_tp, char* id);
    void schedule(py::array_t<int> wk, int backward);
    void init_buffers(torch::Tensor input_tensor, torch::Tensor output_tensor);
    void print_tensor(torch::Tensor tensor, int if_print_content = 1);
    void free_buffers();
    int get_world_size(){return nrank;}
    py::array_t<int> get_buffer_szs();
    int get_rank(){return rank;}
    int get_server_n(){return server_n;}
    int get_hidden_sz(){return hidden_sz;}
    void all_to_all();
    void all_to_all_2(torch::Tensor input_tensor, torch::Tensor output_tensor, torch::Tensor send_tensor, torch::Tensor lbsend_tensor, torch::Tensor lbrecv_tensor,torch::Tensor cros1_tensor, torch::Tensor cros2_tensor, torch::Tensor rstr_tensor);
    void all_to_all_v(torch::Tensor input_tensor, torch::Tensor output_tensor, py::array_t<int> input_splits,  py::array_t<int> output_splits);
    ~flash_t();
};


ncclDataType_t get_dtype(int type){
    switch (type){
        case 0: return ncclInt8;
        case 1: return ncclUint8;
        case 2: return ncclInt32;
        case 3: return ncclUint32;
        case 4: return ncclInt64;
        case 5: return ncclUint64;
        case 6: return ncclFloat16;
        case 7: return ncclFloat32;
        case 8: return ncclFloat64;
        case 9: return ncclBfloat16;
        case 10: return ncclFp8E4M3;
        case 11: return ncclFp8E5M2;
        default: return ncclInt8;
    }
}


flash_t::flash_t(int this_rank, int total_rank, int s_n, int g_n, int h_sz, int d_tp, char* id){
    ncclUniqueId commID;
    for (uint i = 0 ; i < NCCL_UNIQUE_ID_BYTES; i++){
        commID.internal[i] = id[i];
    }
    rank = this_rank;
    nrank = total_rank;
    server_n = s_n;
    gpu_n = g_n;
    NCCLCHECK(ncclCommInitRank(&comm[0], nrank, commID, rank));
    NCCLCHECK(ncclCommSplit(comm[0], 0, rank, &comm[1], NULL));

    RCCLCHECK(hipStreamCreateWithFlags(&stream[0], hipStreamNonBlocking));
    RCCLCHECK(hipStreamCreateWithFlags(&stream[1], hipStreamNonBlocking));

    scheduler = NULL;
    buf_ptrs = NULL;
    hidden_sz = h_sz;
    dtype = get_dtype(d_tp);

    workload = new uint64_t[nrank * nrank];
    std::cout << "RANK " << rank << "/" << nrank << " with s_n = " << server_n << " , g_n = " << gpu_n << " initialize FLASH successfully!" << std::endl;
}


flash_t::~flash_t(){
    for (int i = 1; i >= 0; i --){
        // std::cout << "cleaning flash communicator " << i << std::endl;
        NCCLCHECK(ncclCommFinalize(comm[i]));
        NCCLCHECK(ncclCommDestroy(comm[i]));
        RCCLCHECK(hipStreamDestroy(stream[i]));
    }
    if (scheduler){
        std::cout << "cleaning schedulder" << std::endl;
        free_global_scheduler(scheduler);
        delete scheduler;
    }

    if (buf_ptrs){
        std::cout << "cleaning buffers" << std::endl;
        free_buffers();
        delete buf_ptrs;
    }
    delete[] workload;
    std::cout << "RANK "<< rank << " free FLASH successfully!" << std::endl;
}


py::array_t<int> flash_t::get_buffer_szs(){

    auto result = py::array_t<int>(6);
    py::buffer_info buf = result.request();
    int *ptr = static_cast<int *>(buf.ptr);
    ptr[0] = scheduler -> buff_parameter -> sendbuff_total_sz;
    ptr[1] = scheduler -> buff_parameter -> lbsend_total_sz;
    ptr[2] = scheduler -> buff_parameter -> lbrecv_total_sz;
    ptr[3] = scheduler -> buff_parameter -> crosbuff_total_sz;
    ptr[4] = scheduler -> buff_parameter -> crosbuff_total_sz;
    ptr[5] = scheduler -> buff_parameter -> rstrbuff_total_sz;
    return result;
}


void flash_t::schedule(py::array_t<int> wk, int backward){
    // workload is in terms of tokens
    auto w_buf = wk.request();
    int * wld_ptr = (int *)(w_buf.ptr);
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            workload[i * nrank + j] = (backward == 1) ? wld_ptr[j * nrank + i] : wld_ptr[i * nrank + j];
        }
    }
    // if (rank == 0) print_matrix(workload, nrank, nrank);
    if (server_n > 1){
        if (scheduler == NULL){
            scheduler = new struct GlobalScheduler;
            init_global_scheduler(scheduler, server_n, gpu_n, workload, rank);
            run_scheduler(scheduler);
        }else{
            update_global_scheduler(scheduler, workload);
            run_scheduler(scheduler);
        }
    }
}


void flash_t::print_tensor(torch::Tensor tensor, int if_print_content){
    std::cout << "Dtype: " << tensor.dtype() << std::endl
            << "Layout: " << tensor.layout() << std::endl
            << "Device: "<< tensor.device().type() << std::endl
            << "Address: " <<  tensor.data_ptr() << std::endl
            << "Dim: " <<  tensor.dim() << std::endl;
    if (if_print_content != 0){
        uint ndim = tensor.dim();
        uint element_n = 0;
        for (uint i = 0; i < ndim; i++){
            element_n = MAX(1, element_n) * tensor.size(i);
        }
        std:: cout << "Element#: " << element_n << std::endl;
        void * host_buff = malloc(tensor.element_size() * element_n);
        RCCLCHECK(hipMemcpy(host_buff, tensor.data_ptr(), tensor.element_size() * element_n, hipMemcpyDeviceToHost));
        for (uint i = 0; i < element_n; i++){
            float * tensor_host = (float *)host_buff;
            std::cout << tensor_host[i] << " ";
        }
        std::cout << std::endl;
    }
}


void flash_t::init_buffers(torch::Tensor input_tensor, torch::Tensor output_tensor){
    RCCLCHECK(hipSetDevice(rank % gpu_n));
    void * input = input_tensor.data_ptr();
    void * output = output_tensor.data_ptr();
    if (buf_ptrs){
        std::cout <<"cleaning buff ptrs" << std::endl;
        free_buffers();
        delete buf_ptrs;
    }
    void * sendbuff = NULL, * lbsend = NULL, * lbrecv = NULL, * crosbuff1 = NULL, *crosbuff2 = NULL, * rstrbuff = NULL;

    uint data_type_size = ncclTypeSize(dtype);
    RCCLCHECK(hipMalloc((void **)&sendbuff, scheduler -> buff_parameter -> sendbuff_total_sz * data_type_size * hidden_sz));
    if (scheduler -> buff_parameter -> lbsend_total_sz) {
        RCCLCHECK(hipMalloc((void **)&lbsend, scheduler -> buff_parameter -> lbsend_total_sz * data_type_size * hidden_sz));
    }
    if (scheduler -> buff_parameter -> lbrecv_total_sz) RCCLCHECK(hipMalloc((void **)&lbrecv, scheduler -> buff_parameter -> lbrecv_total_sz * data_type_size * hidden_sz));
    if (scheduler -> buff_parameter -> crosbuff_total_sz) {
        RCCLCHECK(hipMalloc((void **)&crosbuff1, scheduler -> buff_parameter -> crosbuff_total_sz * data_type_size * hidden_sz));
        RCCLCHECK(hipMalloc((void **)&crosbuff2, scheduler -> buff_parameter -> crosbuff_total_sz * data_type_size * hidden_sz));
    }
    if (scheduler -> buff_parameter -> rstrbuff_total_sz) RCCLCHECK(hipMalloc((void **)&rstrbuff, scheduler -> buff_parameter -> rstrbuff_total_sz * data_type_size * hidden_sz));

    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s < server_n; s++){
                uint64_t disp = scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                uint64_t sz = scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                uint64_t offset = scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s];
                uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                RCCLCHECK(hipMemcpy((char*)lbsend + disp * data_type_size * hidden_sz,
                                    (char*)input + (scheduler -> buff_parameter -> inputbuff_disp[dst_gpu_global_id] + offset) * data_type_size * hidden_sz,
                                    sz * data_type_size * hidden_sz,
                                    hipMemcpyDeviceToDevice));
            }
        }
    }

    uint local_rank_id = rank % gpu_n;
    for (uint i = 0; i < gpu_n * server_n; i++){
        uint64_t disp = scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_disp[local_rank_id];
        uint64_t sz = scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_sz[local_rank_id];
        uint64_t offset = scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_offset[local_rank_id];
        RCCLCHECK(hipMemcpy((char*)sendbuff + disp * data_type_size * hidden_sz,
                            (char*)input + (scheduler -> buff_parameter -> inputbuff_disp[i] + offset) * data_type_size * hidden_sz,
                            sz * data_type_size * hidden_sz,
                            hipMemcpyDeviceToDevice));
    }
    RCCLCHECK(hipDeviceSynchronize());
    buf_ptrs = new struct flash_buffer_ptr_t;
    buf_ptrs -> sendbuff = sendbuff;
    buf_ptrs -> recvbuff = output;
    buf_ptrs -> lbsend = lbsend;
    buf_ptrs -> lbrecv = lbrecv;
    buf_ptrs -> crosbuff1 = crosbuff1;
    buf_ptrs -> crosbuff2 = crosbuff2;
    buf_ptrs -> rstrbuff = rstrbuff;
}


void flash_t::free_buffers(){
    if (buf_ptrs){
        RCCLCHECK(hipFree(buf_ptrs->sendbuff));
        if (buf_ptrs->lbsend) RCCLCHECK(hipFree(buf_ptrs->lbsend));
        if (buf_ptrs->lbrecv) RCCLCHECK(hipFree(buf_ptrs->lbrecv));
        if (buf_ptrs->crosbuff1) RCCLCHECK(hipFree(buf_ptrs->crosbuff1));
        if (buf_ptrs->crosbuff2) RCCLCHECK(hipFree(buf_ptrs->crosbuff2));
        if (buf_ptrs->rstrbuff) RCCLCHECK(hipFree(buf_ptrs->rstrbuff));
        delete buf_ptrs;
        buf_ptrs = NULL;
    }

}


ncclResult_t
fanoutAllToAll(void * sendbuff, void * recvbuff, int h_sz, int * input_splits, int * output_splits, uint nrank, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    NCCLCHECK(ncclGroupStart());
    uint64_t input_disp = 0, output_disp = 0;
    for (uint i = 0; i < nrank; i++){
        NCCLCHECK(ncclSend(
            (char *)sendbuff + input_disp * ncclTypeSize(datatype),
            input_splits[i] * h_sz,
            datatype,
            i,
            comm,
            stream
        ));
        input_disp += input_splits[i] * h_sz;
        NCCLCHECK(ncclRecv(
            (char *)recvbuff + output_disp * ncclTypeSize(datatype),
            output_splits[i] * h_sz,
            datatype,
            i,
            comm,
            stream
        ));
        output_disp += output_splits[i] * h_sz;
    }
    NCCLCHECK(ncclGroupEnd());
    RCCLCHECK(hipStreamSynchronize(stream));
    return ncclSuccess;
}


void flash_t::all_to_all_v(torch::Tensor input_tensor, torch::Tensor output_tensor, py::array_t<int> input_splits,  py::array_t<int> output_splits){
    // input_splits and out_splits are in terms of token
    auto input_splits_buf = input_splits.request();
    auto output_splits_buf = output_splits.request();
    at::hip::HIPStream myStream = at::hip::getCurrentHIPStream();
    NCCLCHECK(fanoutAllToAll(
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        hidden_sz,
        (int*)input_splits_buf.ptr,
        (int*)output_splits_buf.ptr,
        nrank,
        dtype,
        comm[0],
        myStream)
    );
}

void flash_t::all_to_all_2(torch::Tensor input_tensor, torch::Tensor output_tensor, torch::Tensor send_tensor, torch::Tensor lbsend_tensor, torch::Tensor lbrecv_tensor,torch::Tensor cros1_tensor, torch::Tensor cros2_tensor, torch::Tensor rstr_tensor){

    assert(scheduler -> buff_parameter -> sendbuff_total_sz <= send_tensor.size(0));
    assert(scheduler -> buff_parameter -> recvbuff_total_sz <= output_tensor.size(0));
    assert(scheduler -> buff_parameter -> crosbuff_total_sz <= cros1_tensor.size(0));
    assert(scheduler -> buff_parameter -> crosbuff_total_sz <= cros2_tensor.size(0));
    assert(scheduler -> buff_parameter -> rstrbuff_total_sz <= rstr_tensor.size(0));
    assert(scheduler -> buff_parameter -> lbsend_total_sz <= lbsend_tensor.size(0));
    assert(scheduler -> buff_parameter -> lbrecv_total_sz <= lbrecv_tensor.size(0));

    if (scheduler -> buff_parameter -> recvbuff_total_sz > output_tensor.size(0)){
        std::cout << "schedule recv: " << scheduler -> buff_parameter -> recvbuff_total_sz <<" , output tensor: " <<  output_tensor.size(0) << " x " <<  output_tensor.size(1) << std::endl;
    }

    at::hip::HIPStream myStream = at::hip::getCurrentHIPStream();
    uint data_type_size = ncclTypeSize(dtype);
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
        for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
            for (uint s = 0; s < server_n; s++){
                uint64_t disp = scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                uint64_t sz = scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                uint64_t offset = scheduler -> buff_parameter -> inputbuff_disp[dst_gpu_global_id] + scheduler -> buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s];
                RCCLCHECK(hipMemcpyWithStream((char*)lbsend_tensor.data_ptr() + disp * data_type_size * hidden_sz,
                                    (char*)input_tensor.data_ptr() + offset * data_type_size * hidden_sz,
                                    sz * data_type_size * hidden_sz,
                                    hipMemcpyDeviceToDevice,
                                    myStream));
            }
        }
    }


    uint local_rank_id = rank % gpu_n;
    for (uint i = 0; i < gpu_n * server_n; i++){
        uint64_t disp = scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_disp[local_rank_id];
        uint64_t sz = scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_sz[local_rank_id];
        uint64_t offset = scheduler -> buff_parameter -> inputbuff_disp[i] + scheduler -> buff_parameter -> sendbuff_region[i].src_gpu_offset[local_rank_id];
        RCCLCHECK(hipMemcpyWithStream((char*)send_tensor.data_ptr() + disp * data_type_size * hidden_sz,
                    (char*)input_tensor.data_ptr() + offset * data_type_size * hidden_sz,
                    sz * data_type_size * hidden_sz,
                    hipMemcpyDeviceToDevice,
                    myStream));
    }

    NCCLCHECK(fastAllToAll(
        send_tensor.data_ptr(),
        output_tensor.data_ptr(),
        lbsend_tensor.data_ptr(),
        lbrecv_tensor.data_ptr(),
        cros1_tensor.data_ptr(),
        cros2_tensor.data_ptr(),
        rstr_tensor.data_ptr(),
        scheduler->gpu_sched,
        hidden_sz,
        dtype,
        comm,
        myStream,
        stream[0]));
}

void flash_t::all_to_all(){
    // Do alltoall
    at::hip::HIPStream myStream = at::hip::getCurrentHIPStream();
    NCCLCHECK(fastAllToAll(
        buf_ptrs->sendbuff,
        buf_ptrs->recvbuff,
        buf_ptrs->lbsend,
        buf_ptrs->lbrecv,
        buf_ptrs->crosbuff1,
        buf_ptrs->crosbuff2,
        buf_ptrs->rstrbuff,
        scheduler->gpu_sched,
        hidden_sz,
        dtype,
        comm,
        myStream,
        stream[0]));
}

PYBIND11_MODULE(FlashAllToAll, m) {
    m.doc() = "pybind11 plugin"; // optional module docstring
    py::class_<flash_t>(m, "flash_t")
        .def(py::init<int, int, int, int, int, int, char*>())
        .def("schedule", &flash_t::schedule)
        .def("get_world_size", &flash_t::get_world_size)
        .def("get_hidden_sz", &flash_t::get_hidden_sz)
        .def("get_buffer_szs", &flash_t::get_buffer_szs)
        .def("get_rank", &flash_t::get_rank)
        .def("get_server_n", &flash_t::get_server_n)
        .def("print_tensor", &flash_t::print_tensor)
        .def("all_to_all", &flash_t::all_to_all)
        .def("all_to_all_v", &flash_t::all_to_all_v)
        .def("all_to_all_2", &flash_t::all_to_all_2)
        .def("init_buffers", &flash_t::init_buffers)
        .def("free_buffers", &flash_t::free_buffers);
}