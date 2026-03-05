#include "fast_alltoall/flash_alltoall_nvshmem.h"
#include <cmath>
#include <iostream>

__forceinline__ __device__ void st_uint64_volatile(uint64_t *flag_addr, uint64_t flag) {
  asm volatile("st.volatile.global.u64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint64_t ld_uint64_volatile(uint64_t *flag_addr) {
  uint64_t flag;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ uint64_t ld_uint64_acquire(uint64_t *flag_addr) {
  uint64_t flag;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ void st_uint64_release(uint64_t *flag_addr, uint64_t flag) {
  asm volatile("st.release.sys.global.u64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint64_t add_uint64_release(uint64_t *addr, uint64_t val) {
  uint64_t flag;
  asm volatile("atom.release.sys.global.add.u64 %0, [%1], %2;" : "=l"(flag) : "l"(addr), "l"(val));
  return flag;
}

__forceinline__ __device__ void atomicSet64(uint64_t* addr, uint64_t val) {
    asm volatile("atom.exch.gpu.b64 %0, [%1], %2;"
                 : "=l"(val) : "l"(addr), "l"(val));
}

__forceinline__ __device__ uint64_t atomicAdd64(uint64_t* addr, uint64_t val) {
    uint64_t old;
    asm volatile ("atom.add.gpu.u64 %0, [%1], %2;"
                  : "=l"(old)
                  : "l"(addr), "l"(val)
                  : "memory");
    return old;
}

__forceinline__ __device__ long long atomicSub64(uint64_t* addr, uint64_t val) {
    long long old;
    asm volatile("atom.add.gpu.u64 %0, [%1], %2;"
                 : "=l"(old)                // output: old value
                 : "l"(addr), "l"(-val)     // inputs: address, -val
                 : "memory");
    return old; // return the value BEFORE subtraction
}

struct kernel_sz_t kernel_sz_for_workload(uint warp_n_per_block, uint block_n, uint TX_BURST){
    uint32_t sm_n = get_sm_count();      // block is mapped to sm, so get number of sms first
    uint32_t actual_block_n = std::min(sm_n, block_n);
    dim3 Gdim(actual_block_n, 1, 1), Bdim(warp_n_per_block * THREAD_N_PER_WARP, 1, 1);
    struct kernel_sz_t k_sz = {
        .grid_dim = Gdim,   // grid number is decided by cuda, only needs to specify grid size
        .block_dim = Bdim,
        .warp_n_per_block = warp_n_per_block,
        .block_n = actual_block_n,
        .min_tx_sz_at_a_time = warp_n_per_block * THREAD_N_PER_WARP * sizeof(int4) * TX_BURST,   // each block sends TX_BURST x [data size of each threads sending sizeof(int4)] at a time
    };
    return k_sz;
}

struct shared_memory_t init_share_memory(void * dev_mem_ptr, uint rank_n_per_node){
    cudaIpcMemHandle_t local_data_handle, * remote_data_handle;
    CUDACHECK(cudaIpcGetMemHandle(&local_data_handle, dev_mem_ptr));
    remote_data_handle = (cudaIpcMemHandle_t *) malloc(sizeof(cudaIpcMemHandle_t) * rank_n_per_node);
    void ** host_remote_data_ptr = (void **) malloc(sizeof(void *) * rank_n_per_node);
    void ** dev_remote_data_ptr;
    CUDACHECK(cudaMalloc(&dev_remote_data_ptr, sizeof(void *) * rank_n_per_node));
    struct shared_memory_t smem = {
        .local_data = dev_mem_ptr,
        .remote_data = dev_remote_data_ptr,
        .host_remote_data = host_remote_data_ptr,
        .local_handle = local_data_handle,
        .remote_handle = remote_data_handle
    };
    return smem;
}


void free_share_memory(struct shared_memory_t  * smem){
    // release resources related to remote ranks
    free(smem -> host_remote_data);
    free(smem -> remote_handle);
    CUDACHECK(cudaFree(smem -> remote_data));
}

// --------------------------------------------------------------------
//              Baseline: Fanout Algorithm
//---------------------------------------------------------------------

#if VERIFY_BUFFER == 1
void verify_intra_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n_per_node, uint rank_n){
    void * host_recvbuff = malloc(bufs -> out_total_sz);
    memset(host_recvbuff, 0,  bufs -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, bufs -> buffer_out, bufs -> out_total_sz, cudaMemcpyDeviceToHost));
    for (uint i = 0; i < rank_n_per_node; i++){
        uint src_rank = (this_rank / rank_n_per_node) * rank_n_per_node + i;
        printf("Intra verify: [RANK %u - from Rank %u] verifying from %lu to %lu (sz: %lu B)- correctness: %u\n", this_rank, src_rank, bufs->out_disp_per_rank[src_rank], bufs->out_disp_per_rank[src_rank] + bufs->out_sz_per_rank[src_rank], bufs->out_sz_per_rank[src_rank], (0 == memcmp((char*)host_recvbuff + bufs->out_disp_per_rank[src_rank], (char*)bufs->buffer_verify + bufs->out_disp_per_rank[src_rank], bufs->out_sz_per_rank[src_rank])));
    }
    free(host_recvbuff);
}

void verify_inter_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n_per_node, uint rank_n){
    void * host_recvbuff = malloc(bufs -> out_total_sz);
    memset(host_recvbuff, 0,  bufs -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, bufs -> buffer_out, bufs -> out_total_sz, cudaMemcpyDeviceToHost));
    printf("Inter verify: [RANK %u] verifying from %lu to %lu (sz: %lu B) - correctness: %u\n", this_rank, bufs->intra_recv_sz, bufs->out_total_sz,  bufs->out_total_sz - bufs->intra_recv_sz, (0 == memcmp((char*)host_recvbuff +  bufs->intra_recv_sz, (char*)bufs->buffer_verify + bufs->intra_recv_sz, bufs->out_total_sz - bufs->intra_recv_sz)));
    free(host_recvbuff);
}

void verify_alltoall_buffers(struct intra_then_inter_data_buffer_t * bufs, uint this_rank, uint rank_n){
    void * host_recvbuff = malloc(bufs -> out_total_sz);
    memset(host_recvbuff, 0,  bufs -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, bufs -> buffer_out, bufs -> out_total_sz, cudaMemcpyDeviceToHost));
    printf("[RANK %u] verifying from 0 to %lu (sz: %lu B)- correctness: %u\n", this_rank, bufs -> out_total_sz, bufs -> out_total_sz, (0 == memcmp((char*)host_recvbuff, (char*)bufs->buffer_verify, bufs -> out_total_sz)));
    free(host_recvbuff);
}

#endif

struct intra_then_inter_data_buffer_t init_data_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n){
    uint64_t * send_offset_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recv_offset_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * send_sz_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recv_sz_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recv_offset_at_remote_buffer = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);

    uint server_id = this_rank / rank_n_per_node;
    uint local_rank_id = this_rank % rank_n_per_node;
    uint64_t send_sz = 0, recv_sz = 0, remote_recv_offset = 0;
    // first is intra-node data
    for (uint i = 0; i < rank_n_per_node; i++) {
        uint global_rank = server_id * rank_n_per_node + i;
        send_offset_per_rank[global_rank] = send_sz;
        send_sz_per_rank[global_rank] = workload_in_byte[this_rank * rank_n + global_rank];
        send_sz += workload_in_byte[this_rank * rank_n + global_rank];
        assert(workload_in_byte[this_rank * rank_n + global_rank] % sizeof(int4) == 0);
        recv_offset_per_rank[global_rank] = recv_sz;
        recv_sz_per_rank[global_rank] = workload_in_byte[global_rank * rank_n + this_rank];
        recv_sz += workload_in_byte[global_rank * rank_n + this_rank];
        assert(workload_in_byte[global_rank * rank_n + this_rank] % sizeof(int4) == 0);
        // for remote ranks, the offset in the recv buffer
        remote_recv_offset = 0;
        for (uint src = 0; src < local_rank_id; src++){
            uint src_global_id = server_id * rank_n_per_node + src;
            remote_recv_offset += workload_in_byte[src_global_id * rank_n + global_rank];
        }
        recv_offset_at_remote_buffer[global_rank] = remote_recv_offset;
    }
    uint64_t intra_send_size = send_sz, intra_recv_size = recv_sz;
    // then is inter-node data
    for (uint r = 0; r < rank_n; r++){
        if (r / rank_n_per_node == server_id){
            continue;
        }
        send_offset_per_rank[r] = send_sz;
        send_sz_per_rank[r] = workload_in_byte[this_rank * rank_n + r];
        send_sz += workload_in_byte[this_rank * rank_n + r];
        assert(workload_in_byte[this_rank * rank_n + r] % sizeof(int4) == 0);
        recv_offset_per_rank[r] = recv_sz;
        recv_sz_per_rank[r] = workload_in_byte[r * rank_n + this_rank];
        recv_sz += workload_in_byte[r * rank_n + this_rank];
        assert(workload_in_byte[r * rank_n + this_rank] % sizeof(int4) == 0);
        // for remote ranks, the offset in the recv buffer
        remote_recv_offset = 0;
        uint remote_server_id = r / rank_n_per_node;
        for (uint src = 0; src < rank_n_per_node; src++){
            uint src_global_id = remote_server_id * rank_n_per_node + src;
            remote_recv_offset += workload_in_byte[src_global_id * rank_n + r];
        }
        for (uint src = 0; src < this_rank; src++){
            if (src / rank_n_per_node == remote_server_id){
                continue;
            }
            remote_recv_offset += workload_in_byte[src * rank_n + r];
        }
        recv_offset_at_remote_buffer[r] = remote_recv_offset;
    }
    // allocate and initialize send/recv buffer
    void * sendbuff, * recvbuff;

    CUDACHECK(cudaMalloc((void **)&sendbuff, send_sz));
    CUDACHECK(cudaMalloc((void **)&recvbuff, recv_sz));
    CUDACHECK(cudaMemset(sendbuff, 0, send_sz));
    CUDACHECK(cudaMemset(recvbuff, 0, recv_sz));

    uint64_t int_num = send_sz / sizeof(int32_t);
    int32_t * host_sendbuff = new int32_t[int_num];
    memset(host_sendbuff, 0, int_num * sizeof(int32_t));
    for (uint i = 0; i < rank_n; i++){
        for (uint64_t j = 0; j < send_sz_per_rank[i] / sizeof(int32_t); j++){
            int32_t unique_data = ((this_rank & 0xff) << 24) + ((i & 0xff) << 16) + (j & 0xffff);
            host_sendbuff[send_offset_per_rank[i] / sizeof(int32_t) + j] = unique_data;
        }
    }
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)sendbuff, (void *) host_sendbuff,  send_sz));

#if VERIFY_BUFFER == 1
    void * verifybuff = malloc(recv_sz);
    memset(verifybuff, 0, recv_sz);
    for (uint i = 0; i < rank_n; i++){
        for (uint64_t j = 0; j < recv_sz_per_rank[i] / sizeof(int32_t); j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((this_rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [recv_offset_per_rank[i] / sizeof(int32_t) + j] = unique_data;
        }
    }
#endif

    // init share memory using IPC
    struct shared_memory_t shared_recv_buffer = init_share_memory(recvbuff, rank_n_per_node);

    CUDACHECK(cudaDeviceSynchronize());
    delete[] host_sendbuff;
    // finish intialization
    struct intra_then_inter_data_buffer_t buf = {
        .buffer_in = sendbuff,
        .in_total_sz = send_sz,
        .in_sz_per_rank = send_sz_per_rank,
        .in_disp_per_rank = send_offset_per_rank,
        .buffer_out = recvbuff,
        .out_total_sz = recv_sz,
        .out_sz_per_rank = recv_sz_per_rank,
        .out_disp_per_rank = recv_offset_per_rank,
        .out_disp_at_remote_buffer = recv_offset_at_remote_buffer,
        .share_mem = {
            .recv_buffer = shared_recv_buffer,
        },
        .intra_send_sz = intra_send_size,
        .intra_recv_sz = intra_recv_size,
#if VERIFY_BUFFER == 1
        .buffer_verify = verifybuff
#endif
    };
    return buf;

}

void free_data_buffer(struct intra_then_inter_data_buffer_t * buf){
    CUDACHECK(cudaFree(buf -> buffer_in));
    CUDACHECK(cudaFree(buf -> buffer_out));
    free_share_memory(&buf -> share_mem.recv_buffer);
    free(buf -> in_disp_per_rank);
    free(buf -> in_sz_per_rank);
    free(buf -> out_disp_per_rank);
    free(buf -> out_sz_per_rank);
#if VERIFY_BUFFER == 1
    free(buf -> buffer_verify);
#endif
}

struct fanout_data_buffer_t init_fanout_data_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n){
    uint64_t * send_offset_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recv_offset_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * send_sz_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recv_sz_per_rank = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);

    uint64_t send_sz = 0, recv_sz = 0;
    for (uint r = 0; r < rank_n; r++){
        send_offset_per_rank[r] = send_sz;
        send_sz_per_rank[r] = workload_in_byte[this_rank * rank_n + r];
        send_sz += workload_in_byte[this_rank * rank_n + r];
        assert(workload_in_byte[this_rank * rank_n + r] % sizeof(int32_t) == 0);
        recv_offset_per_rank[r] = recv_sz;
        recv_sz_per_rank[r] = workload_in_byte[r * rank_n + this_rank];
        recv_sz += workload_in_byte[r * rank_n + this_rank];
        assert(workload_in_byte[r * rank_n + this_rank] % sizeof(int32_t) == 0);
    }
    // send/recv buffer
    void * sendbuff, * recvbuff;

    CUDACHECK(cudaMalloc((void **)&sendbuff, send_sz));
    CUDACHECK(cudaMalloc((void **)&recvbuff, recv_sz));
    CUDACHECK(cudaMemset(sendbuff, 0, send_sz));
    CUDACHECK(cudaMemset(recvbuff, 0, recv_sz));

    uint64_t int_num = send_sz / sizeof(int32_t);
    int32_t * host_sendbuff = new int32_t[int_num];
    memset(host_sendbuff, 0, int_num * sizeof(int32_t));
    for (uint i = 0; i < rank_n; i++){
        for (uint64_t j = 0; j < send_sz_per_rank[i] / sizeof(int32_t); j++){
            int32_t unique_data = ((this_rank & 0xff) << 24) + ((i & 0xff) << 16) + (j & 0xffff);
            host_sendbuff[send_offset_per_rank[i] / sizeof(int32_t) + j] = unique_data;
        }
    }
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)sendbuff, (void *) host_sendbuff,  send_sz));
    CUDACHECK(cudaDeviceSynchronize());
    delete[] host_sendbuff;

#if VERIFY_BUFFER == 1
    void * verifybuff = malloc(recv_sz);
    memset(verifybuff, 0, recv_sz);
    for (uint i = 0; i < rank_n; i++){
        for (uint64_t j = 0; j < recv_sz_per_rank[i] / sizeof(int32_t); j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((this_rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [recv_offset_per_rank[i] / sizeof(int32_t) + j] = unique_data;
        }
    }
#endif

    // init share memory using IPC
    struct shared_memory_t shared_recv_buffer = init_share_memory(recvbuff, rank_n_per_node);

    // finish intialization
    struct fanout_data_buffer_t buf = {
        .buffer_in = sendbuff,
        .in_total_sz = send_sz,
        .in_sz_per_rank = send_sz_per_rank,
        .in_disp_per_rank = send_offset_per_rank,
        .buffer_out = recvbuff,
        .out_total_sz = recv_sz,
        .out_sz_per_rank = recv_sz_per_rank,
        .out_disp_per_rank = recv_offset_per_rank,
        .share_mem = {
            .recv_buffer = shared_recv_buffer,
        },
#if VERIFY_BUFFER == 1
        .buffer_verify = verifybuff
#endif
    };
    return buf;
}


void free_fanout_data_buffer(struct fanout_data_buffer_t  * buf){
    CUDACHECK(cudaFree(buf -> buffer_in));
    CUDACHECK(cudaFree(buf -> buffer_out));
    free_share_memory(&buf -> share_mem.recv_buffer);
    free(buf -> in_disp_per_rank);
    free(buf -> in_sz_per_rank);
    free(buf -> out_disp_per_rank);
    free(buf -> out_sz_per_rank);
#if VERIFY_BUFFER == 1
    free(buf -> buffer_verify);
#endif
}


struct fanout_nvshmem_buffer_t init_fanout_nvshmem_buffer(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n){
    uint64_t max_send_sz = 0, max_recv_sz = 0;
    for (uint i = 0; i < rank_n; i++){
        uint64_t send_sz = 0, recv_sz = 0;
        for (uint j = 0; j < rank_n; j++){
            send_sz += workload_in_byte[i * rank_n + j];
            recv_sz += workload_in_byte[j * rank_n + i];
        }
        max_send_sz = MAX(max_send_sz, send_sz);
        max_recv_sz = MAX(max_recv_sz, recv_sz);
    }
    printf("Rank %u, send buffer: %lu, recv buffer: %lu\n", this_rank, max_send_sz, max_recv_sz);
    uint8_t * send_buffer = (uint8_t *) nvshmem_malloc(max_send_sz);
    uint8_t * recv_buffer = (uint8_t *) nvshmem_malloc(max_recv_sz);
    uint64_t * sync_signal = (uint64_t *)nvshmem_calloc(rank_n * block_n, sizeof(uint64_t));
    struct fanout_nvshmem_buffer_t buffs = {
        .send_buffer = send_buffer,
        .recv_buffer = recv_buffer,
        .sync_signal = sync_signal,
    };
    return buffs;
}


void free_fanout_nvshmem(struct fanout_nvshmem_buffer_t * buf){
    nvshmem_free(buf -> send_buffer);
    nvshmem_free(buf -> recv_buffer);
    nvshmem_free(buf -> sync_signal);
}

struct internode_transfer_params_t init_internode_transfer_params(struct intra_then_inter_data_buffer_t * data_buf, uint this_rank, uint rank_n_per_node, uint rank_n){
    uint64_t * sender_send_disp = (uint64_t *)  malloc(sizeof(uint64_t) * rank_n);
    uint64_t * sender_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * sender_recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    uint64_t * recver_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n);
    for (uint r = 0; r < rank_n; r++){
        //receiver logic
        recver_transfer_sz[r] = data_buf -> out_sz_per_rank[r];
        sender_send_disp[r] = data_buf -> in_disp_per_rank[r];     // offset in the sender buffer for each rank
        sender_transfer_sz[r] = data_buf -> in_sz_per_rank[r];
        sender_recv_disp[r] = data_buf -> out_disp_at_remote_buffer[r];;     // offset in the receiver buffer for each rank
    }

     uint64_t * d_sender_send_disp, * d_sender_transfer_sz, * d_sender_recv_disp, * d_recver_transfer_sz;
    CUDACHECK(cudaMalloc(&d_sender_send_disp, sizeof(uint64_t) * rank_n ));
    CUDACHECK(cudaMalloc(&d_sender_transfer_sz, sizeof(uint64_t) * rank_n ));
    CUDACHECK(cudaMalloc(&d_sender_recv_disp, sizeof(uint64_t) * rank_n ));
    CUDACHECK(cudaMalloc(&d_recver_transfer_sz, sizeof(uint64_t) * rank_n ));


    CUDACHECK(cudaMemcpy(d_sender_send_disp, sender_send_disp, sizeof(uint64_t) * rank_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_transfer_sz, sender_transfer_sz, sizeof(uint64_t) * rank_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_recv_disp, sender_recv_disp, sizeof(uint64_t) * rank_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recver_transfer_sz, recver_transfer_sz, sizeof(uint64_t) * rank_n, cudaMemcpyHostToDevice));


    free(sender_send_disp);
    free(sender_transfer_sz);
    free(sender_recv_disp);
    free(recver_transfer_sz);
    struct internode_transfer_params_t params = {
        .sender_send_disp = d_sender_send_disp,
        .sender_transfer_sz = d_sender_transfer_sz,
        .sender_recv_disp = d_sender_recv_disp,
        .recver_transfer_sz = d_recver_transfer_sz,
    };
    return params;
}


void free_internode_transfer_params(struct internode_transfer_params_t * inter_params){
    CUDACHECK(cudaFree(inter_params -> sender_send_disp));
    CUDACHECK(cudaFree(inter_params -> sender_transfer_sz));
    CUDACHECK(cudaFree(inter_params -> sender_recv_disp));
    CUDACHECK(cudaFree(inter_params -> recver_transfer_sz));
}


struct intranode_transfer_params_t init_intranode_transfer_params(struct intra_then_inter_data_buffer_t * data_buf, uint this_rank, uint rank_n_per_node, uint rank_n, struct kernel_sz_t * ksz){
    uint block_n = ksz -> block_n;
    uint64_t * sender_send_disp = (uint64_t *)  malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint64_t * sender_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint64_t * sender_recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint64_t * recver_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint64_t * recver_recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint64_t * intra_send_round_n = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);

    uint server_id = this_rank / rank_n_per_node;
    uint local_rank_id = this_rank % rank_n_per_node;

    uint64_t recv_offset = 0, send_offset = 0;
    for (uint r = 0; r < rank_n_per_node; r++){
        uint target_rank = server_id * rank_n_per_node + r;
        //receiver logic
        uint64_t recv_offset_from_rank = data_buf -> out_disp_per_rank[target_rank];
        uint64_t recv_sz_from_rank = data_buf -> out_sz_per_rank[target_rank];
        uint64_t recv_sz_per_block = CEIL_DIV(recv_sz_from_rank, block_n);
        uint64_t already_recv_sz = 0;
        for (uint b = 0; b < block_n; b++){
            uint64_t actual_recv_sz_per_block = std::min(recv_sz_per_block, recv_sz_from_rank - already_recv_sz);
            recver_transfer_sz[r * block_n + b] = actual_recv_sz_per_block;
            recver_recv_disp[r * block_n + b] = recv_offset_from_rank + already_recv_sz;
            already_recv_sz += actual_recv_sz_per_block;
        }
        // sender logic
        send_offset = data_buf -> in_disp_per_rank[target_rank];
        recv_offset = data_buf -> out_disp_at_remote_buffer[target_rank];
        uint64_t send_sz_to_rank = data_buf->in_sz_per_rank[target_rank];
        uint64_t send_sz_per_block = CEIL_DIV(send_sz_to_rank, block_n);
        uint64_t already_sent_sz = 0;
        for (uint b = 0; b < block_n; b++){
            uint64_t actual_send_sz_per_block = std::min(send_sz_per_block, send_sz_to_rank - already_sent_sz);
            sender_send_disp[r * block_n + b] = send_offset + already_sent_sz;     // offset in the sender buffer for each block
            sender_recv_disp[r * block_n + b] = recv_offset + already_sent_sz;     // offset in the receiver buffer for each block
            sender_transfer_sz[r * block_n + b] = actual_send_sz_per_block;        // block total transfer size
            already_sent_sz += actual_send_sz_per_block;
        }
    }   
    // calculate intra-node alltoall round#
    for (uint r = 0; r < rank_n_per_node; r++){
        for (uint b = 0; b < block_n; b++){
            intra_send_round_n[r * block_n + b] = (sender_transfer_sz[r * block_n + b] + ksz -> min_tx_sz_at_a_time - 1) / ksz -> min_tx_sz_at_a_time;
        }
    }
    // print parameters
    if (this_rank == 0){
        for (uint r = 0; r < rank_n_per_node; r++){
            printf("rank %u\n", r);
            for (uint b = 0; b < block_n; b++){
                printf("\tblock-%u, sender_send_offset: %lu, sender_recv_offset: %lu, sender_sz: %lu, recver_recv_offset: %lu, recver_sz: %lu, send rounds: %lu\n", b, sender_send_disp[r * block_n +b], sender_recv_disp[r * block_n +b], sender_transfer_sz[r * block_n + b], recver_recv_disp[r * block_n + b], recver_transfer_sz[r * block_n + b], intra_send_round_n[r * block_n + b]);
            }
        }
    }

    uint64_t * d_sender_send_disp, * d_sender_transfer_sz, * d_sender_recv_disp, * d_recver_recv_disp, * d_recver_transfer_sz, * d_intra_send_round_n;
    CUDACHECK(cudaMalloc(&d_sender_send_disp, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_sender_transfer_sz, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_sender_recv_disp, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_recver_transfer_sz, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_recver_recv_disp, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_intra_send_round_n, sizeof(uint64_t) * rank_n_per_node * block_n));


    CUDACHECK(cudaMemcpy(d_sender_send_disp, sender_send_disp, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_transfer_sz, sender_transfer_sz, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_recv_disp, sender_recv_disp, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recver_transfer_sz, recver_transfer_sz, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recver_recv_disp, recver_recv_disp, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_intra_send_round_n, intra_send_round_n, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));


    free(sender_send_disp);
    free(sender_transfer_sz);
    free(sender_recv_disp);
    free(recver_transfer_sz);
    free(recver_recv_disp);
    free(intra_send_round_n);
    struct intranode_transfer_params_t  params = {
        .sender_send_disp = d_sender_send_disp,
        .sender_transfer_sz = d_sender_transfer_sz,
        .sender_recv_disp = d_sender_recv_disp,
        .recver_transfer_sz = d_recver_transfer_sz,
        .recver_recv_disp = d_recver_recv_disp,
        .intra_round_n = d_intra_send_round_n,
    };
    return params;
}


void free_intranode_transfer_params(struct intranode_transfer_params_t * intra_params){
    CUDACHECK(cudaFree(intra_params -> sender_send_disp));
    CUDACHECK(cudaFree(intra_params -> sender_transfer_sz));
    CUDACHECK(cudaFree(intra_params -> sender_recv_disp));
    CUDACHECK(cudaFree(intra_params -> recver_transfer_sz));
    CUDACHECK(cudaFree(intra_params -> recver_recv_disp));
    CUDACHECK(cudaFree(intra_params -> intra_round_n));
}

struct fanout_buffer_params_t init_fanout_params(uint64_t * workload_in_byte, uint this_rank, uint rank_n_per_node, uint rank_n, struct kernel_sz_t * ksz){
    uint block_n = ksz -> block_n;
    uint64_t * sender_send_disp = (uint64_t *)  malloc(sizeof(uint64_t) * rank_n * block_n);
    uint64_t * sender_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n * block_n);
    uint64_t * sender_recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * rank_n * block_n);
    uint64_t * recver_transfer_sz = (uint64_t *) malloc(sizeof(uint64_t) * rank_n * block_n);
    uint64_t * recver_recv_disp = (uint64_t *) malloc(sizeof(uint64_t) * rank_n * block_n);

    uint64_t send_offset = 0;
    for (uint r = 0; r < rank_n; r++){
        //receiver logic
        uint64_t recv_offset = 0;
        for (uint src_r = 0; src_r < r; src_r ++){
            recv_offset += workload_in_byte[src_r * rank_n + this_rank];
        }
        uint64_t recv_sz_from_rank = workload_in_byte[r * rank_n + this_rank];
        uint64_t recv_sz_per_block = CEIL_DIV(recv_sz_from_rank, block_n);
        uint64_t already_recv_sz = 0;
        for (uint b = 0; b < block_n; b++){
            uint64_t actual_recv_sz_per_block = std::min(recv_sz_per_block, recv_sz_from_rank - already_recv_sz);
            recver_transfer_sz[r * block_n + b] = actual_recv_sz_per_block;
            recver_recv_disp[r * block_n + b] = recv_offset + already_recv_sz;
            already_recv_sz += actual_recv_sz_per_block;
        }
        // sender logic
        recv_offset = 0;
        for (uint src_r = 0; src_r < this_rank; src_r ++){
            recv_offset += workload_in_byte[src_r * rank_n + r];
        }
        uint64_t send_sz_to_rank = workload_in_byte[this_rank * rank_n + r];
        uint64_t send_sz_per_block = CEIL_DIV(send_sz_to_rank, block_n);
        uint64_t already_sent_sz = 0;
        for (uint b = 0; b < block_n; b++){
            uint64_t actual_send_sz_per_block = std::min(send_sz_per_block, send_sz_to_rank - already_sent_sz);
            sender_send_disp[r * block_n + b] = send_offset + already_sent_sz;     // offset in the sender buffer for each block
            sender_recv_disp[r * block_n + b] = recv_offset + already_sent_sz;     // offset in the receiver buffer for each block
            sender_transfer_sz[r * block_n + b] = actual_send_sz_per_block;        // block total transfer size
            already_sent_sz += actual_send_sz_per_block;
        }
        send_offset += send_sz_to_rank;
    }   

    // calculate intra-node alltoall round#
    uint64_t * intra_round_n = (uint64_t *) malloc(sizeof(uint64_t) * rank_n_per_node * block_n);
    uint32_t server_id = this_rank / rank_n_per_node;
    for (uint r = 0; r < rank_n_per_node; r++){
        uint32_t global_rank = server_id * rank_n_per_node + r;
        for (uint b = 0; b < block_n; b++){
            intra_round_n[r * block_n + b] = (sender_transfer_sz[global_rank * block_n + b] + ksz -> min_tx_sz_at_a_time - 1) / ksz -> min_tx_sz_at_a_time;
        }
    }

    //calculate inter-node alltoall around#
    uint64_t * inter_send_round_n = (uint64_t *) malloc (sizeof(uint64_t) * rank_n * block_n);
    uint64_t * inter_recv_round_n = (uint64_t *) malloc (sizeof(uint64_t) * rank_n * block_n);
    for (uint r = 0; r < rank_n; r++){
        for (uint b = 0; b < block_n; b++){
            inter_send_round_n[r * block_n + b] = (sender_transfer_sz[r * block_n + b] + ksz -> min_tx_sz_at_a_time - 1) / ksz -> min_tx_sz_at_a_time;
            inter_recv_round_n[r * block_n + b] = (recver_transfer_sz[r * block_n + b] + ksz -> min_tx_sz_at_a_time - 1) / ksz -> min_tx_sz_at_a_time;
        }
    }

    // print parameters
    if (this_rank == 0){
        for (uint r = 0; r < rank_n; r++){
            printf("rank %u\n", r);
            for (uint b = 0; b < block_n; b++){
                printf("\tblock-%u, sender_send_offset: %lu, sender_recv_offset: %lu, sender_sz: %lu, recver_recv_offset: %lu, recver_sz: %lu, send rounds: %lu, recv rounds: %lu\n", b, sender_send_disp[r * block_n +b], sender_recv_disp[r * block_n +b], sender_transfer_sz[r * block_n + b], recver_recv_disp[r * block_n + b], recver_transfer_sz[r * block_n + b], inter_send_round_n[r * block_n + b], inter_recv_round_n[r * block_n + b]);
            }
        }
    }

    // copy metadata from host to device
    uint64_t * d_sender_send_disp, * d_sender_transfer_sz, * d_sender_recv_disp, * d_recver_recv_disp, * d_recver_transfer_sz, * d_intra_round_n, * d_inter_send_round_n, * d_inter_recv_round_n;
    CUDACHECK(cudaMalloc(&d_sender_send_disp, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_sender_transfer_sz, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_sender_recv_disp, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_recver_transfer_sz, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_recver_recv_disp, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_intra_round_n, sizeof(uint64_t) * rank_n_per_node * block_n));
    CUDACHECK(cudaMalloc(&d_inter_send_round_n, sizeof(uint64_t) * rank_n * block_n));
    CUDACHECK(cudaMalloc(&d_inter_recv_round_n, sizeof(uint64_t) * rank_n * block_n));


    CUDACHECK(cudaMemcpy(d_sender_send_disp, sender_send_disp, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_transfer_sz, sender_transfer_sz, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sender_recv_disp, sender_recv_disp, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recver_transfer_sz, recver_transfer_sz, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recver_recv_disp, recver_recv_disp, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_intra_round_n, intra_round_n, sizeof(uint64_t) * rank_n_per_node * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_inter_send_round_n, inter_send_round_n, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_inter_recv_round_n, inter_recv_round_n, sizeof(uint64_t) * rank_n * block_n, cudaMemcpyHostToDevice));


    free(sender_send_disp);
    free(sender_transfer_sz);
    free(sender_recv_disp);
    free(recver_transfer_sz);
    free(recver_recv_disp);
    free(intra_round_n);
    free(inter_send_round_n);
    free(inter_recv_round_n);
    struct fanout_buffer_params_t  params = {
        .sender_send_disp = d_sender_send_disp,
        .sender_transfer_sz = d_sender_transfer_sz,
        .sender_recv_disp = d_sender_recv_disp,
        .recver_transfer_sz = d_recver_transfer_sz,
        .recver_recv_disp = d_recver_recv_disp,
        .intra_round_n = d_intra_round_n,
        .inter_send_round_n = d_inter_send_round_n,
        .inter_recv_round_n = d_inter_recv_round_n
    };
    return params;
}

void free_fanout_params(struct fanout_buffer_params_t * params){
    CUDACHECK(cudaFree(params -> sender_send_disp));
    CUDACHECK(cudaFree(params -> sender_transfer_sz));
    CUDACHECK(cudaFree(params -> sender_recv_disp));
    CUDACHECK(cudaFree(params -> recver_transfer_sz));
    CUDACHECK(cudaFree(params -> recver_recv_disp));
    CUDACHECK(cudaFree(params -> intra_round_n));
    CUDACHECK(cudaFree(params -> inter_send_round_n));
    CUDACHECK(cudaFree(params -> inter_recv_round_n));
}


__device__ __forceinline__ void device_default_memcpy(
    void * dst_buffer,
    const void * src_buffer,
    const uint64_t sz
){
    // use the whole thread blocks for memory copy
    const int4 * src_vec = (const int4 *) src_buffer;
    int4* dst_vec = (int4*) dst_buffer;
    const uint64_t vec_n = sz / sizeof(int4);
    const uint64_t tail = vec_n * sizeof(int4);
    for(uint64_t j = threadIdx.x + blockIdx.x * blockDim.x; j < vec_n; j +=  gridDim.x * blockDim.x){ 
        dst_vec[j] = src_vec[j];
    }
    if (threadIdx.x + blockIdx.x * blockDim.x == 0){
        const uint8_t * src_ptr = (const uint8_t *) src_buffer;
        uint8_t * dst_ptr = (uint8_t *) dst_buffer;
        for (uint i = tail; i < sz; i ++){
            dst_ptr[i] = src_ptr[i];
        }
    }
}


__global__ void spreadout_alltoallv_internode_kernel(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    const uint32_t rank_n,
    // nvshmem memory for RDMA data exchange
    uint8_t * send_buffer,
    uint8_t * recv_buffer,
    uint64_t * sync_signal,  
    // metadata for internode transfer
    const uint64_t * inter_sender_send_disp,
    const uint64_t * inter_sender_transfer_sz,
    const uint64_t * inter_sender_recv_disp,
    const uint64_t * inter_recver_transfer_sz
){
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    
    const uint32_t server_id = this_rank / local_rank_n;
    const uint32_t local_rank_id = this_rank % local_rank_n;
    const uint32_t server_n = rank_n / local_rank_n;
    const uint32_t inter_node_rank_n = rank_n - local_rank_n;

    if (warp_id == 0){
        //use warp 0 in block 0 to do inter-node transfer
        for (uint step = 1; step < server_n; step ++){
            const uint32_t dst_server_id = (server_id + step) % server_n;
            const uint32_t src_server_id = (server_id + server_n - step) % server_n;
            for (uint j = 0; j < local_rank_n; j ++){
                const uint32_t send_rank_id = dst_server_id * local_rank_n + (local_rank_id + j) % local_rank_n;
                const uint32_t recv_rank_id = src_server_id * local_rank_n + (local_rank_id + local_rank_n - j) % local_rank_n;
                const uint64_t send_offset = __ldg(&inter_sender_send_disp[send_rank_id]);
                const uint64_t recv_offset = __ldg(&inter_sender_recv_disp[send_rank_id]);
                const int64_t send_sz = __ldg(&inter_sender_transfer_sz[send_rank_id]);
                nvshmemx_putmem_signal_nbi_warp(
                    recv_buffer + recv_offset,
                    send_buffer + send_offset,
                    send_sz,
                    &sync_signal[this_rank],
                    send_sz,
                    NVSHMEM_SIGNAL_ADD,
                    send_rank_id
                );
            }   
        }
        nvshmem_quiet();
    }else if (warp_id == 1){
        for (uint i = lane_id; i < inter_node_rank_n; i += THREAD_N_PER_WARP){
            const uint32_t src_rank = ((server_id + 1)* local_rank_n + i) % rank_n;
            const int64_t recv_sz = __ldg(&inter_recver_transfer_sz[src_rank]);
            nvshmem_uint64_wait_until(&sync_signal[src_rank], NVSHMEM_CMP_EQ, recv_sz);
            sync_signal[src_rank] = 0;
        }
    }
}


__global__ void spreadout_alltoallv_intranode_kernel(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    // metadata for intranode transfer
    const uint64_t * intra_sender_send_disp,
    const uint64_t * intra_sender_transfer_sz,
    const uint64_t * intra_sender_recv_disp,
    const uint64_t * intra_recver_transfer_sz,
    const uint64_t * intra_recver_recv_disp,
    const uint64_t * intra_send_round_n,
    const uint64_t  min_tx_sz_at_a_time
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint peer_local_id = (local_rank_id + step) % local_rank_n;
        const uint64_t intra_round_n = intra_send_round_n[peer_local_id * block_n + block_id];
        const uint64_t send_offset = __ldg(&intra_sender_send_disp[peer_local_id * block_n + block_id]);
        const uint64_t send_sz_this_rank = __ldg(&intra_sender_transfer_sz[peer_local_id * block_n + block_id]);
        const uint64_t recv_offset = __ldg(&intra_sender_recv_disp[peer_local_id * block_n + block_id]);
        for (uint64_t round = 0; round < intra_round_n; round ++){
            const uint64_t tx_offset = round * min_tx_sz_at_a_time;
            const uint64_t send_offset_this_round = send_offset + tx_offset;
            const uint64_t recv_offset_this_round = recv_offset + tx_offset;
            const uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
            const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
            const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
            const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
            const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;

            // STEP1: copy data from input buffer to shared buffer
            const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
            const int4* input_vec = (const int4*)input_buf;
            uint8_t * share_buf = (uint8_t *) shared_memory;
            int4* share_vec = (int4*) share_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                }  
            }
            // STEP2: copy data from shared buffer to output buffer
            uint8_t * output_buf = (uint8_t *) remote_buffer[peer_local_id] + recv_offset_this_round;
            int4* output_vec = (int4*)output_buf;

            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                }  
            }

            __syncthreads(); // Ensure shared memory write is complete before output
        }
        cooperative_groups::this_grid().sync();
    }
}

__global__ void p2p_kernel(
    void * buffer_out,
    const void * buffer_in,
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    const uint32_t rank_n,
    // nvshmem mem
    uint8_t * send_buffer,
    uint8_t * recv_buffer,
    uint64_t * sync_signal,  
    // metadata required for data exchange
    const uint64_t * sender_send_disp,
    const uint64_t * sender_transfer_sz,
    const uint64_t * sender_recv_disp,
    const uint64_t * recver_transfer_sz,
    const uint64_t * recver_recv_disp,
    const uint64_t * inter_send_round_n,
    const uint64_t * inter_recv_round_n,
    // control TX burst size
    const uint64_t  min_tx_sz_at_a_time,
    const uint32_t src_server,
    const uint32_t dst_server
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n =  blockDim.x / THREAD_N_PER_WARP;

    if (this_rank / local_rank_n == src_server){
        const uint32_t dst_rank = dst_server * local_rank_n + this_rank % local_rank_n;
        const uint64_t round_n = __ldg(&inter_send_round_n[dst_rank * block_n + block_id]);
        const uint64_t send_offset = __ldg(&sender_send_disp[dst_rank * block_n + block_id]);
        const uint64_t recv_offset =  __ldg(&sender_recv_disp[dst_rank * block_n + block_id]);
        const int64_t send_sz_to_rank = __ldg(&sender_transfer_sz[dst_rank * block_n + block_id]);

        // for (uint64_t round = 0; round < round_n; round ++){
        //     const int64_t tx_offset = round * min_tx_sz_at_a_time;
        //     const uint64_t send_offset_this_round = send_offset + tx_offset;
        //     const uint64_t recv_offset_this_round = recv_offset + tx_offset;
        //     const int64_t actual_send_sz_to_rank = MIN(min_tx_sz_at_a_time, send_sz_to_rank - tx_offset);
        //     // // for sender
        //     // const uint64_t send_total_vec_n = actual_send_sz_to_rank / sizeof(int4);
        //     // const uint64_t send_full_warp_n = send_total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
        //     // const uint64_t send_vec_tail_start = send_full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
        //     // const uint64_t send_remainder_vec_n = send_total_vec_n - send_vec_tail_start;

        //     // // STEP1: copy data from input buffer to nvshmem send buffer
        //     // const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
        //     // const int4* input_vec = (const int4*)input_buf;
        //     // uint8_t * send_buf = send_buffer + send_offset_this_round;
        //     // int4* send_vec = (int4*) send_buf;

        //     // for(uint64_t j = warp_id; j < send_full_warp_n; j += warp_n){ 
        //     //     const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
        //     //     int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
        //     //     #pragma unroll
        //     //     for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
        //     //         input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
        //     //     }
        //     //     #pragma unroll
        //     //     for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
        //     //         send_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
        //     //     }
        //     // }
        //     // if (warp_id == warp_n - 1){
        //     //     for (uint i = lane_id; i < send_remainder_vec_n; i += THREAD_N_PER_WARP) {
        //     //         send_vec[send_vec_tail_start + i] = input_vec[send_vec_tail_start + i];   // each thread does int4 transmission
        //     //     }  
        //     // }
        //     // __syncthreads();
            
        // }

        if (warp_id == 0){
            nvshmemx_putmem_signal_nbi_warp(
                recv_buffer,
                send_buffer,
                send_sz_to_rank,
                &sync_signal[this_rank * block_n + block_id],
                send_sz_to_rank * local_rank_n,
                NVSHMEM_SIGNAL_ADD,
                dst_rank
            ); 
        }
  

        // const uint32_t rounds = send_sz_to_rank / min_tx_sz_at_a_time;
        // const uint32_t times_per_warp = CEIL_DIV(rounds, warp_n);
        // const uint32_t warp_offset = warp_id * times_per_warp;
        // const uint32_t actual_times_this_warp = MIN(times_per_warp, rounds - warp_offset);
        // for (uint i = 0; i < actual_times_this_warp; i++){
        //     nvshmemx_putmem_signal_nbi_warp(
        //         recv_buffer + recv_offset + warp_offset + i * min_tx_sz_at_a_time,
        //         send_buffer + send_offset + warp_offset + i * min_tx_sz_at_a_time,
        //         min_tx_sz_at_a_time,
        //         &sync_signal[src_rank * block_n + block_id],
        //         min_tx_sz_at_a_time,
        //         NVSHMEM_SIGNAL_ADD,
        //         dst_rank
        //     );   
        // }
    }
    nvshmem_quiet();
    // STEP2: copy data from recv buffer to output buffer
    // if(threadIdx.x == 0){
    //     nvshmem_uint64_wait_until(&sync_signal[recv_rank_id * block_n + block_id], NVSHMEM_CMP_EQ, copy_sz_from_rank);
    // }
    // __syncthreads();
    // uint8_t * recv_buf = (uint8_t *) recv_buffer + copy_offset;
    // int4* recv_vec = (int4*)recv_buf;
    // uint8_t * output_buf = (uint8_t *) buffer_out + copy_offset;
    // int4* output_vec = (int4*)output_buf;

    // // for receiver
    // const uint64_t recv_total_vec_n = copy_sz_from_rank / sizeof(int4);
    // const uint64_t recv_full_warp_n = recv_total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP);
    // const uint64_t recv_vec_tail_start = recv_full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
    // const uint64_t recv_remainder_vec_n = recv_total_vec_n - recv_vec_tail_start;

    // for(uint64_t j = warp_id; j < recv_full_warp_n; j += warp_n){ 
    //     const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
    //     int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
    //     #pragma unroll
    //     for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
    //         output_reg[k] = recv_vec[thread_offset + k * THREAD_N_PER_WARP];
    //     }
    //     #pragma unroll
    //     for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
    //         output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
    //     }
    // }
    // if (warp_id == warp_n - 1){
    //     for (uint i = lane_id; i < recv_remainder_vec_n; i += THREAD_N_PER_WARP) {
    //         output_vec[recv_vec_tail_start + i] = recv_vec[recv_vec_tail_start + i];   // each thread does int4 transmission
    //     }  
    // }
    
}

__global__ void intranode_alltoallv_kernel(
    // void * buffer_out,
    const void * buffer_in,
    const uint32_t this_rank,
    const uint32_t rank_n_per_node,
    // const uint32_t rank_n,
    // share mem
    // uint64_t * local_already_send_sz,
    // uint64_t ** remote_already_send_sz,
    void ** remote_buffer,
    // nvshmem mem
    // uint8_t * send_buffer,
    // uint8_t * recv_buffer,
    // uint64_t * sync_signal,  
    // metadata required for data exchange
    const uint64_t * sender_send_disp,
    const uint64_t * sender_transfer_sz,
    const uint64_t * sender_recv_disp,
    // const uint64_t * recver_transfer_sz,
    // const uint64_t * recver_recv_disp,
    const uint64_t * intra_round_n,
    // control TX burst size
    const uint64_t  min_tx_sz_at_a_time
){

    // use shared memory within thread block cluster
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    // const uint32_t thread_n_per_block = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n =  blockDim.x / THREAD_N_PER_WARP;

    const uint32_t server_id = this_rank / rank_n_per_node;
    const uint32_t local_rank_id = this_rank % rank_n_per_node;

    // alltoallv within the same host
    for (uint step = 0; step < rank_n_per_node; step ++){
        const uint peer_local_id = (local_rank_id + step) % rank_n_per_node;
        const uint peer_global_id = server_id * rank_n_per_node + (local_rank_id + step) % rank_n_per_node;
        const uint64_t intra_send_round_n = intra_round_n[peer_local_id * block_n + block_id];
        const uint64_t send_offset = __ldg(&sender_send_disp[peer_global_id * block_n + block_id]);
        const uint64_t send_sz_this_rank = __ldg(&sender_transfer_sz[peer_global_id * block_n + block_id]);
        const uint64_t recv_offset = __ldg(&sender_recv_disp[peer_global_id * block_n + block_id]);
        for (uint64_t round = 0; round < intra_send_round_n; round ++){
            const uint64_t tx_offset = round * min_tx_sz_at_a_time;
            const uint64_t send_offset_this_round = send_offset + tx_offset;
            const uint64_t recv_offset_this_round = recv_offset + tx_offset;
            const uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
            const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
            const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
            const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
            const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;

            // STEP1: copy data from input buffer to shared buffer
            const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
            const int4* input_vec = (const int4*)input_buf;
            uint8_t * share_buf = (uint8_t *) shared_memory;
            int4* share_vec = (int4*) share_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                }  
            }
            // STEP2: copy data from shared buffer to output buffer
            uint8_t * output_buf = (uint8_t *) remote_buffer[peer_local_id] + recv_offset_this_round;
            int4* output_vec = (int4*)output_buf;

            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                }  
            }
        }
        cooperative_groups::this_grid().sync();
    }
    // cooperative_groups::this_grid().sync();
}


__global__ void local_p2p_kernel(
    void * buffer_out,
    const void * buffer_in,
    const uint32_t this_rank,
    const uint32_t rank_n_per_node,
    const uint32_t rank_n,
    // share mem
    uint64_t * local_already_send_sz,
    uint64_t ** remote_already_send_sz,
    void ** remote_buffer,
    // nvshmem mem
    uint8_t * send_buffer,
    uint8_t * recv_buffer,
    uint64_t * sync_signal,  
    // metadata required for data exchange
    const uint64_t * sender_send_disp,
    const uint64_t * sender_transfer_sz,
    const uint64_t * sender_recv_disp,
    const uint64_t * recver_transfer_sz,
    const uint64_t * recver_recv_disp,
    const uint64_t * intra_round_n,
    // control TX burst size
    const uint64_t  min_tx_sz_at_a_time,
    const uint32_t src_rank,
    const uint32_t dst_rank
){

    // use shared memory within thread block cluster
    extern __shared__ __align__(16) uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    const uint32_t thread_n_per_block = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n =  blockDim.x / THREAD_N_PER_WARP;

    const uint32_t server_id = this_rank / rank_n_per_node;
    const uint32_t local_rank_id = this_rank % rank_n_per_node;
    const uint32_t local_rank_n = rank_n_per_node;

    const uint32_t dst_local_rank = dst_rank % rank_n_per_node;
    // p2p within the same host
    if (this_rank == src_rank){
            const uint64_t intra_send_round_n = intra_round_n[dst_local_rank * block_n + block_id];
            const uint global_rank = dst_rank;
            const uint64_t send_offset = __ldg(&sender_send_disp[global_rank * block_n + block_id]);
            const uint64_t send_sz_this_rank = __ldg(&sender_transfer_sz[global_rank * block_n + block_id]);
            const uint64_t recv_offset = __ldg(&sender_recv_disp[global_rank * block_n + block_id]);
            uint64_t tx_offset = 0;
            for (uint64_t round = 0; round < intra_send_round_n; round ++){
                const uint64_t send_offset_this_round = send_offset + tx_offset;
                const uint64_t recv_offset_this_round = recv_offset + tx_offset;
                const uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);

                const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
                uint8_t * share_buf = (uint8_t *) shared_memory;

                const uint64_t num_full_chunks = actual_send_sz_this_rank / sizeof(int4);
                const uint64_t recv_warp_n = num_full_chunks / TX_UNROLL_FACTOR / THREAD_N_PER_WARP;   // each warp sends 4 * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
                const int4* input_vec = (const int4*)input_buf;
                int4* share_vec = (int4*) share_buf;

                // copy data from global src memory to shared memory first
                for(uint64_t j = warp_id; j < recv_warp_n; j += warp_n){ 
                    int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread
                    const uint64_t start_pos = j * TX_UNROLL_FACTOR * THREAD_N_PER_WARP + lane_id;
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        input_reg[k] = input_vec[start_pos + k * THREAD_N_PER_WARP];
                    }
                    
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        share_vec[start_pos + k * THREAD_N_PER_WARP] = input_reg[k];
                    }
                }
                // copy data from shared memory to dst memory
                uint8_t * output_buf = (uint8_t *) remote_buffer[dst_local_rank] + recv_offset_this_round;
                int4* output_vec = (int4*)output_buf;

                for(uint64_t j = warp_id; j < recv_warp_n; j += warp_n){ 
                    int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread
                    const uint64_t start_pos = j * TX_UNROLL_FACTOR * THREAD_N_PER_WARP + lane_id;
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_reg[k] = share_vec[start_pos + k * THREAD_N_PER_WARP];
                    }
                    
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_vec[start_pos + k * THREAD_N_PER_WARP] = output_reg[k];
                    }
                }
                // use the last warp to handle remainder chunks
                const uint64_t tail_start = recv_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
                const uint64_t remainder_chunks_n = num_full_chunks - tail_start;
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_chunks_n; i += THREAD_N_PER_WARP) {
                        output_vec[tail_start + i] = share_vec[tail_start + i];   // each thread does int4 transmission
                    }  
                }
                tx_offset += actual_send_sz_this_rank;
                // __syncthreads();
            }
    }
    // cooperative_groups::this_grid().sync();
}

__global__ void fanout_alltoallv_kernel(
    void * buffer_out,
    const void * buffer_in,
    const uint32_t this_rank,
    const uint32_t rank_n_per_node,
    const uint32_t rank_n,
    // nvshmem mem
    uint8_t * send_buffer,
    uint8_t * recv_buffer,
    uint64_t * sync_signal,  
    // metadata required for data exchange
    const uint64_t * sender_send_disp,
    const uint64_t * sender_transfer_sz,
    const uint64_t * sender_recv_disp,
    const uint64_t * recver_transfer_sz,
    const uint64_t * recver_recv_disp
){


// Send Buffer
// |     dst rank 0         |       dst rank1      |       dst rank 2       |      ...      |     dst rank n      |
// |  b0  | b1   | ... | bn |  b0| b1   | ... | bn |  b0  | b1   | ... | bn |
// |TX| ..|

//ROUND 0: dst 0: TX, dst1: TX, .... dst n:TX


    uint64_t TX_BURST_PER_BLOCK = 16;
    uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    uint32_t thread_n_per_block = blockDim.x;
    uint32_t block_id = blockIdx.x;
    uint32_t block_n = gridDim.x;
    uint64_t tx_burst_sz_per_block = thread_n_per_block * sizeof(int4) * TX_BURST_PER_BLOCK;  // each thread handles sizeof(int4) amount of data in a single operation at maximum


    // determine the maximum sending/recving rounds
    uint64_t max_send_round = 0;
    for (uint r = 0; r < rank_n; r++){
        uint64_t send_rounds = (sender_transfer_sz[r * block_n + block_id] + tx_burst_sz_per_block - 1) / tx_burst_sz_per_block;
        if (send_rounds > max_send_round) max_send_round = send_rounds;
    }

    for (uint64_t round = 0; round < max_send_round; round ++){
        for (uint r = 0; r < rank_n; r++){
            uint64_t send_sz_this_rank = sender_transfer_sz[r * block_n + block_id];
            uint64_t tx_offset = round * tx_burst_sz_per_block;
            if (tx_offset < send_sz_this_rank){
                uint64_t actual_send_sz_this_rank = MIN(tx_burst_sz_per_block, send_sz_this_rank - tx_offset);
                uint64_t send_offset = sender_send_disp[r * block_n + block_id] + tx_offset;

                uint8_t * input_buf = (uint8_t *) buffer_in + send_offset;
                uint8_t * send_buf = (uint8_t *) send_buffer + send_offset;

                uint64_t num_full_chunks = actual_send_sz_this_rank / sizeof(int4);
                uint64_t remainder_bytes = actual_send_sz_this_rank % sizeof(int4);

                int4* input_vec = (int4*)input_buf;
                int4* output_vec = (int4*)send_buf;

                // does the memcpy for each thread in this block
                for (int i = threadIdx.x; i < num_full_chunks; i += thread_n_per_block) {
                    output_vec[i] = input_vec[i];   // each thread does int4 transmission
                }
                if (remainder_bytes > 0){
                    int tail_start = num_full_chunks * sizeof(int4);
                    if (threadIdx.x == thread_n_per_block - 1) {
                        for (uint64_t j = 0; j < remainder_bytes; j ++) {
                            send_buf[tail_start + j] = input_buf[tail_start + j];
                        }
                    }
                }
                __syncthreads();
                // now do the sending when memcpy is completed at this round
                // The first warp initiates sending
                if (warp_id == 0) {
                    uint8_t * recv_buf = recv_buffer + (sender_recv_disp[r * block_n + block_id]) + tx_offset;
                    nvshmemx_putmem_signal_nbi_warp(
                        recv_buf,
                        send_buf,
                        actual_send_sz_this_rank,
                        &sync_signal[this_rank * block_n + block_id],
                        actual_send_sz_this_rank,
                        NVSHMEM_SIGNAL_ADD,
                        r
                    );
                }
            }
        }
    }

    // sync to ensure that all recv are done.
    for (size_t i = threadIdx.x; i < rank_n; i += thread_n_per_block) {
        uint64_t recv_sz_from_rank = __ldg(&recver_transfer_sz[i * block_n + block_id]);
        nvshmem_uint64_wait_until(&sync_signal[i * block_n + block_id], NVSHMEM_CMP_EQ, recv_sz_from_rank);
    }
    __syncthreads();
    // memcpy to the output buffer
    for (uint r = 0; r < rank_n; r++){
        uint64_t recv_offset_from_rank = __ldg(&recver_recv_disp[r * block_n + block_id]);
        uint64_t recv_sz_from_rank = __ldg(&recver_transfer_sz[r * block_n + block_id]);

        uint8_t * output_buf = (uint8_t *) buffer_out + recv_offset_from_rank;
        uint8_t * recv_buf = (uint8_t *) recv_buffer + recv_offset_from_rank;

        uint64_t num_full_chunks = recv_sz_from_rank / sizeof(int4);
        uint64_t remainder_bytes = recv_sz_from_rank % sizeof(int4);

        int4* input_vec = (int4*)recv_buf;
        int4* output_vec = (int4*)output_buf;

        // does the memcpy for each thread in this block
        for (int i = threadIdx.x; i < num_full_chunks; i += thread_n_per_block) {
            output_vec[i] = input_vec[i];   // each thread does int4 transmission
        }

        if (remainder_bytes > 0){
            int tail_start = num_full_chunks * sizeof(int4);
            if (threadIdx.x == thread_n_per_block - 1) {
                for (uint64_t j = 0; j < remainder_bytes; j ++) {
                    output_buf[tail_start + j] = recv_buf[tail_start + j];
                }
            }
        }
    }
}




int launch_fanout_alltoallv(
    struct fanout_data_buffer_t * data_buf,
    uint32_t this_rank,
    uint32_t rank_n_per_node,
    uint32_t rank_n,
    struct fanout_nvshmem_buffer_t * buf,
    struct fanout_buffer_params_t * params,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream
){
    void* args[] = {
        // &data_buf->buffer_out,
        &data_buf->buffer_in,
        &this_rank,
        &rank_n_per_node,
        // &rank_n,
        // &data_buf -> share_mem.already_send_sz.local_data,
        // &data_buf -> share_mem.already_send_sz.remote_data,
        &data_buf -> share_mem.recv_buffer.remote_data,
        // &buf->send_buffer,
        // &buf->recv_buffer,
        // &buf->sync_signal,
        &params->sender_send_disp,
        &params->sender_transfer_sz,
        &params->sender_recv_disp,
        // &params->recver_transfer_sz, 
        // &params->recver_recv_disp,
        &params->intra_round_n,
        &k_sz -> min_tx_sz_at_a_time,
    };
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&intranode_alltoallv_kernel,
            k_sz -> grid_dim,
            k_sz -> block_dim, 
            args,
            k_sz -> min_tx_sz_at_a_time,      // shared memory size per block
            stream
        ));
    return NVSHMEMX_SUCCESS;
}


int launch_alltoallv(
    struct intra_then_inter_data_buffer_t * data_buf,
    uint32_t this_rank,
    uint32_t rank_n_per_node,
    uint32_t rank_n,
    struct fanout_nvshmem_buffer_t * buf,
    struct intranode_transfer_params_t * intra_params,
    struct internode_transfer_params_t * inter_params,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2
){

    void * intra_args[] = {
        &this_rank,
        &rank_n_per_node,
        &data_buf -> share_mem.recv_buffer.remote_data,
        &data_buf->buffer_in,
        &intra_params->sender_send_disp,
        &intra_params->sender_transfer_sz,
        &intra_params->sender_recv_disp,
        &intra_params->recver_transfer_sz,
        &intra_params->recver_recv_disp,
        &intra_params->intra_round_n,
        &k_sz -> min_tx_sz_at_a_time
    };

    void* inter_args[] = {
        &this_rank,
        &rank_n_per_node,
        &rank_n,
        &buf->send_buffer,
        &buf->recv_buffer,
        &buf->sync_signal,
        &inter_params->sender_send_disp,
        &inter_params->sender_transfer_sz,
        &inter_params->sender_recv_disp,
        &inter_params->recver_transfer_sz,
    };
    dim3 inter_grid(1, 1, 1), inter_block(THREAD_N_PER_2WARP, 1, 1);
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&spreadout_alltoallv_internode_kernel,
            inter_grid,
            inter_block, 
            inter_args,
            0,
            stream1
        ));
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&spreadout_alltoallv_intranode_kernel,
            k_sz -> grid_dim,
            k_sz -> block_dim, 
            intra_args,
            k_sz ->min_tx_sz_at_a_time,      // shared memory size per block
            stream2
        ));
    CUDACHECK(cudaStreamSynchronize(stream2));
    CUDACHECK(cudaStreamSynchronize(stream1));
    return NVSHMEMX_SUCCESS;
}

struct flash_buffer_ptr_t init_flash_buffer(struct flash_buffer_sz_params_t * params, uint this_rank, uint rank_n_per_node, uint rank_n, uint block_n, uint data_size){
    // printf("[FLASH RANK %u]: in_sz: %ld, out_sz: %ld, balance_send_sz: %ld, balance_recv_sz: %ld, send_sz: %ld, max_send_sz: %ld, ping-pong_sz: %ld\n", this_rank, params -> in_total_sz, params -> out_total_sz, params -> balance_send_param.total_sz, params -> balance_recv_param.total_sz, params -> send_param.total_sz, params -> send_param.max_total_sz, params -> pingpong_sz);
    // allocate memory
    void * d_buffer_in = NULL, * d_buffer_out = NULL, * d_balance_send = NULL, * d_balance_recv = NULL, * d_recv_complete_signal = NULL, * d_redistribute_complete_signal = NULL;
    CUDACHECK(cudaMalloc((void **)&d_buffer_in, params -> in_total_sz * data_size));
    CUDACHECK(cudaMalloc((void **)&d_buffer_out, params -> out_total_sz * data_size));
    if (params -> balance_send_param.total_sz) {
        CUDACHECK(cudaMalloc((void **)&d_balance_send, params -> balance_send_param.total_sz * data_size));
    }
    uint64_t balance_recv_sz = MAX(1024, params -> balance_recv_param.total_sz * data_size);
    CUDACHECK(cudaMalloc((void **)&d_balance_recv, balance_recv_sz * data_size));
    CUDACHECK(cudaMalloc((void **)&d_recv_complete_signal, sizeof(uint64_t) * rank_n_per_node * block_n * (MAX_TRANSFER_STEP_NUM + 1)));
    CUDACHECK(cudaMalloc((void **)&d_redistribute_complete_signal, sizeof(uint64_t) * block_n * (MAX_TRANSFER_STEP_NUM + 1)));
    CUDACHECK(cudaMemset(d_recv_complete_signal, 0, sizeof(uint64_t) * rank_n_per_node * block_n * (MAX_TRANSFER_STEP_NUM + 1)));
    CUDACHECK(cudaMemset(d_redistribute_complete_signal, 0, sizeof(uint64_t) * block_n * (MAX_TRANSFER_STEP_NUM + 1)));
    struct shared_memory_t shared_out_buffer = init_share_memory(d_buffer_out, rank_n_per_node);
    struct shared_memory_t shared_balance_recv_buffer = init_share_memory(d_balance_recv, rank_n_per_node);
    struct shared_memory_t shared_recv_completion_signal = init_share_memory(d_recv_complete_signal, rank_n_per_node);

    // NVSHMEM
    uint server_n = (rank_n  + rank_n_per_node - 1) / rank_n_per_node;
    uint8_t * send_buffer = (uint8_t *) nvshmem_malloc(params -> send_param.max_total_sz * data_size);
    uint8_t * recv_buffer1 = (uint8_t *) nvshmem_malloc(params -> pingpong_sz * data_size);
    uint8_t * recv_buffer2 = (uint8_t *) nvshmem_malloc(params -> pingpong_sz * data_size);
    uint64_t * sync_signal1 = (uint64_t *) nvshmem_calloc(rank_n, sizeof(uint64_t));
    uint64_t * sync_signal2 = (uint64_t *) nvshmem_calloc(rank_n, sizeof(uint64_t));
    CUDACHECK(cudaMemset(send_buffer, 0, params -> send_param.max_total_sz * data_size));
    CUDACHECK(cudaMemset(recv_buffer1, 0, params -> pingpong_sz * data_size));
    CUDACHECK(cudaMemset(recv_buffer2, 0, params -> pingpong_sz * data_size));
    // credit buffer
    uint64_t * stage_credit =  (uint64_t *) nvshmem_calloc(4 * rank_n, sizeof(uint64_t));

    CUDACHECK(cudaDeviceSynchronize());

#if VERIFY_BUFFER == 1
    void * verifybuff = malloc(params -> out_total_sz * data_size);
    memset(verifybuff, 0, params -> out_total_sz * data_size);
    for (uint i = 0; i < rank_n; i++){
        assert((params->out_disp_per_rank[i] * data_size)% sizeof(int32_t) == 0);
        assert((params->out_sz_per_rank[i] * data_size)% sizeof(int32_t) == 0);
        for (uint64_t j = 0; j < (params->out_sz_per_rank[i] * data_size) / sizeof(int32_t); j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((this_rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [(params->out_disp_per_rank[i] * data_size) / sizeof(int32_t) + j] = unique_data;
        }
    }
#endif

    // initialize memory - buffer in 
    uint8_t * host_buffer_in = (uint8_t *) malloc(params -> in_total_sz * data_size);
    memset(host_buffer_in, 0, params -> in_total_sz * data_size);
    for (uint i = 0; i < rank_n; i++){
        uint64_t disp = params -> in_disp_per_rank[i] * data_size;
        uint64_t sz = params -> in_sz_per_rank[i] * data_size;
        assert(sz % sizeof(int32_t) == 0);
        assert(disp % sizeof(int32_t) == 0);

        for (uint64_t j = 0; j < sz / sizeof(int32_t); j++){
            int32_t unique_data = ((this_rank & 0xff) << 24) + ((i & 0xff) << 16) + (j & 0xffff);
            int32_t * sb = (int32_t *) host_buffer_in;
            sb[disp / sizeof(int32_t) + j] = unique_data;
        }
    }
    CUCHECK(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)d_buffer_in, (void *) host_buffer_in, params -> in_total_sz * data_size));

    //initialize memory - balance send buffer
    uint local_rank_id = this_rank % rank_n_per_node;
    uint8_t * host_lbsend = NULL;
    if (params -> balance_send_param.total_sz > 0){
        host_lbsend = (uint8_t *) malloc(params -> balance_send_param.total_sz * data_size);
        memset(host_lbsend, 0, params -> balance_send_param.total_sz * data_size);
        for (uint local_gpu = 0; local_gpu < rank_n_per_node; local_gpu ++){
            for (uint dst_gpu = 0; dst_gpu < rank_n_per_node; dst_gpu ++){
                for (uint s = 0; s < server_n; s++){
                    uint dst_global_rank = s * rank_n_per_node + dst_gpu;
                    uint64_t disp = params -> balance_send_param.layout[local_gpu].disp[dst_global_rank] * data_size;
                    uint64_t sz = params -> balance_send_param.layout[local_gpu].sz[dst_global_rank] * data_size;
                    uint64_t offset = params -> balance_send_param.layout[local_gpu].data_offset[dst_global_rank] * data_size;
                    uint64_t in_disp = params -> in_disp_per_rank[dst_global_rank] * data_size;
                    CUDACHECK(cudaMemcpy((char*)d_balance_send + disp, (char *)d_buffer_in + in_disp + offset, sz, cudaMemcpyDeviceToDevice));
                    // if (this_rank == 0 && sz > 0) printf("init: buffer-in from %lu to %lu ===> balance-send from %lu to %lu, sz: %lu\n", in_disp + offset, in_disp + offset + sz, disp, disp + sz, sz);
                }
            }
        }
    }

    // if (this_rank == 9){
    //     uint8_t * host_lbsendbuff = (uint8_t *)malloc(params -> balance_send_param.total_sz);
    //     CUDACHECK(cudaMemcpy(host_lbsendbuff, d_balance_send, params -> balance_send_param.total_sz, cudaMemcpyDeviceToHost));
    //     printf("BALANCE SEND BUFFER: \n");
    //     for (uint dst_local_gpu = 0; dst_local_gpu < rank_n_per_node; dst_local_gpu ++){
    //         printf("TO RANK %u: ", dst_local_gpu);
    //         for (uint i = 0; i < rank_n; i++){
    //         if (params -> balance_send_param.sz[dst_local_gpu] > 0){
    //                 for (uint z = 0; z < params -> balance_send_param.layout[dst_local_gpu].sz[i]; z++){
    //                     printf("%02x", host_lbsendbuff[params -> balance_send_param.layout[dst_local_gpu].disp[i] + z]);
    //                 }
    //                 printf("|");
    //             }   
    //         }
    //         printf("\n");
    //     }
    //     free(host_lbsendbuff);
    // }

    // uint local_rank_id = this_rank % rank_n_per_node;
    // uint8_t * host_lbsend = NULL;
    // if (params -> balance_send_param.total_sz > 0){
    //     host_lbsend = (uint8_t *) malloc(params -> balance_send_param.total_sz);
    //     memset(host_lbsend, 0, params -> balance_send_param.total_sz);
    //     for (uint local_gpu = 0; local_gpu < rank_n_per_node; local_gpu ++){
    //         for (uint dst_gpu = 0; dst_gpu < rank_n_per_node; dst_gpu ++){
    //             for (uint s = 0; s < server_n; s++){
    //                 uint dst_global_rank = s * rank_n_per_node + dst_gpu;
    //                 uint64_t disp = params -> balance_send_param.layout[local_gpu].disp[dst_global_rank];
    //                 uint64_t sz = params -> balance_send_param.layout[local_gpu].sz[dst_global_rank];
    //                 uint64_t offset = params -> balance_send_param.layout[local_gpu].data_offset[dst_global_rank];
    //                 assert(sz % sizeof(int32_t) == 0);
    //                 assert(disp % sizeof(int32_t) == 0);
    //                 assert(offset % sizeof(int32_t) == 0);               
    //                 for (uint64_t z = 0; z < sz / sizeof(int32_t); z ++){
    //                     int32_t unique_data =  ((this_rank & 0xff) << 24) + ((dst_global_rank & 0xff) << 16) + ((z + offset / sizeof(int32_t)) & 0xffff);
    //                     int32_t * bb = (int32_t *) host_lbsend;
    //                     bb[disp / sizeof(int32_t) + z] = unique_data;
    //                 }
    //             }
    //         }
    //     }
    //     CUCHECK(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)d_balance_send, (void *) host_lbsend, params -> balance_send_param.total_sz));
    // }

    // initialize memory - send buffer
    uint8_t * host_sendbuff = (uint8_t *) malloc(params -> send_param.total_sz * data_size);
    memset(host_sendbuff, 0, params -> send_param.total_sz * data_size);
    for (uint i = 0; i < rank_n; i++){
        uint64_t disp = params -> send_param.layout[i].src_gpu_disp[local_rank_id] * data_size;
        uint64_t sz = params -> send_param.layout[i].src_gpu_sz[local_rank_id] * data_size;
        uint64_t offset = params -> send_param.layout[i].data_offset[local_rank_id] * data_size;
        uint64_t in_disp = params -> in_disp_per_rank[i] * data_size;
        CUDACHECK(cudaMemcpy((char *)send_buffer + disp, (char *)d_buffer_in + in_disp + offset, sz, cudaMemcpyDeviceToDevice));
        // if (this_rank == 0 && sz > 0) printf("init: buffer-in from %lu to %lu ===> send-buffer from %lu to %lu, sz: %lu\n", in_disp + offset, in_disp + offset + sz, disp, disp + sz, sz);
    }

    // if (this_rank == 0){
    //     uint8_t * host_sendbuff = (uint8_t *)malloc(params -> send_param.total_sz);
    //     CUDACHECK(cudaMemcpy(host_sendbuff, send_buffer, params -> send_param.total_sz, cudaMemcpyDeviceToHost));
    //     printf("SEND BUFFER BEFORE LOAD BALANCING: \n");
    //     for (uint i = 0; i < rank_n; i++){
    //         printf("TO RANK %u: ", i);
    //         if (params -> send_param.sz[i] > 0){
    //             for (uint src_gpu = 0; src_gpu < rank_n_per_node; src_gpu ++){
    //                 for (uint z = 0; z < params -> send_param.layout[i].src_gpu_sz[src_gpu]; z++){
    //                     printf("%02x", host_sendbuff[params -> send_param.layout[i].src_gpu_disp[src_gpu] + z]);
    //                 }
    //                 printf("|");
    //             }   
    //         }
    //         printf("\n");
    //     }
    //     free(host_sendbuff);
    // }

    // uint8_t * host_sendbuff = (uint8_t *) malloc(params -> send_param.total_sz);
    // memset(host_sendbuff, 0, params -> send_param.total_sz);
    // for (uint i = 0; i < rank_n; i++){
    //     uint64_t disp = params -> send_param.layout[i].src_gpu_disp[local_rank_id];
    //     uint64_t sz = params -> send_param.layout[i].src_gpu_sz[local_rank_id];
    //     uint64_t offset = params -> send_param.layout[i].data_offset[local_rank_id];
    //     assert(sz % sizeof(int32_t) == 0);
    //     assert(disp % sizeof(int32_t) == 0);
    //     assert(offset % sizeof(int32_t) == 0);

    //     for (uint64_t j = 0; j < sz / sizeof(int32_t); j++){
    //         int32_t unique_data = ((this_rank & 0xff) << 24) + ((i & 0xff) << 16) + ((j + offset / sizeof(int32_t)) & 0xffff);
    //         int32_t * sb = (int32_t *) host_sendbuff;
    //         sb[disp / sizeof(int32_t) + j] = unique_data;
    //     }
    // }
    // CUCHECK(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)send_buffer, (void *) host_sendbuff, params -> send_param.total_sz));
    

    CUDACHECK(cudaDeviceSynchronize());
    if (params -> balance_send_param.total_sz > 0){
        free(host_lbsend);
    }
    // free(host_sendbuff);
    struct flash_buffer_ptr_t bufs = {
        .buffer_in = d_buffer_in,
        .buffer_out = d_buffer_out,
        .balance_send_buffer = d_balance_send,
        .balance_recv_buffer = d_balance_recv,
        .recv_complete_signal = d_recv_complete_signal,
        .redistribute_complete_signal = d_redistribute_complete_signal,
        .share_mem = {
            .out_buffer = shared_out_buffer,
            .balance_recv_buffer = shared_balance_recv_buffer,
            .recv_complete_signal = shared_recv_completion_signal
        },
        .send_buffer = send_buffer,
        .internode_buffer1 = recv_buffer1,
        .sync_signal1 = sync_signal1,
        .internode_buffer2 = recv_buffer2,
        .sync_signal2 = sync_signal2,
        .credit = stage_credit,
#if VERIFY_BUFFER == 1
        .buffer_verify = verifybuff
#endif
    };
    return bufs;
}


void free_flash_buffer(struct flash_buffer_ptr_t * buf){
    CUDACHECK(cudaFree(buf->buffer_in));
    CUDACHECK(cudaFree(buf->buffer_out));
    if (buf->balance_send_buffer) CUDACHECK(cudaFree(buf->balance_send_buffer));
    CUDACHECK(cudaFree(buf->balance_recv_buffer));
    free_share_memory(&buf -> share_mem.balance_recv_buffer);
    free_share_memory(&buf -> share_mem.out_buffer);
    free_share_memory(&buf -> share_mem.recv_complete_signal);
    nvshmem_free(buf->send_buffer);
    nvshmem_free(buf->internode_buffer1);
    nvshmem_free(buf->internode_buffer2);
    nvshmem_free(buf->sync_signal1);
    nvshmem_free(buf->sync_signal2);
    nvshmem_free(buf->credit);
#if VERIFY_BUFFER == 1
    free(buf->buffer_verify);
#endif
}

// -----------------------------------------------------------------------------
//                                flash kernel
// ----------------------------------------------------------------------------
__forceinline__  __device__ void wait_and_consume_credit(uint64_t* credit, const int buf_idx, const int stage_id, const int this_rank){
  if (threadIdx.x == 0) {
    // Wait until receiver granted this buffer
    nvshmem_uint64_wait_until(&credit[buf_idx], NVSHMEM_CMP_GE, stage_id);
    // Consume the credit (single writer in your 1:1 pairing)
    // asm volatile("membar.gl;" ::: "memory");
  }
  __syncwarp();
}

__global__ void grant_initial_credit_to_peer( uint64_t* peer_credit_sym, const int buf_idx, const int peer_rank, const int stage_id){
    if (threadIdx.x == 0) {
        nvshmemx_signal_op(&peer_credit_sym[buf_idx], stage_id, NVSHMEM_SIGNAL_SET, peer_rank);
    }
}


__global__ void grant_credit_to_peer(uint64_t * sync_signal, uint64_t * redis_complete, struct device_inter_p2p_params_t * p, uint64_t* peer_credit_sym, const int buf_idx, const int peer_rank, const int stage_id, const int this_rank, const uint block_n, uint64_t sync_threshold){
  // Make sure all reads from this buffer are done before granting
//   if(threadIdx.x == 0){
//     printf("RANK %u START granting credit - stage %u - to RANK %u\n", this_rank, stage_id, peer_rank);
//   }
  if (stage_id > 1) {// no need to wait for redistribution completion for stage 1
    for (uint i = threadIdx.x; i < block_n; i += blockDim.x){
        while(true){
            uint64_t val = *((volatile uint64_t*)(&redis_complete[i]));
            if (val == 1){
                redis_complete[i] = 0;
                // atomicSet64(&redis_complete[i], 0);
                break;
            }
        }
    }
    __syncthreads();
  }
//   if(threadIdx.x == 0){
//     printf("RANK %u MID granting credit - stage %u - to RANK %u\n", this_rank, stage_id, peer_rank);
//   }
  if (threadIdx.x == 0) {
    // Set sender's local credit[b] = 1 on the peer
    nvshmem_uint64_wait_until(&sync_signal[p -> src_rank], NVSHMEM_CMP_GE,  p -> recv_sz);
    nvshmemx_signal_op(&peer_credit_sym[buf_idx], stage_id, NVSHMEM_SIGNAL_SET, peer_rank);
    // printf("RANK %u END granting credit - stage %u - to RANK %u\n", this_rank, stage_id, peer_rank);
  }
}

__global__ void flash_internode_p2p(
    const uint32_t this_rank,
    // nvshmem memory for RDMA data exchange
    uint8_t * send_buffer,
    uint8_t * recv_buffer,
    uint64_t * sync_signal,  
    // credit
    uint64_t * credit,
    const uint32_t credit_idx,
    const uint32_t stage_id,
    // metadata for internode transfer
    const struct device_inter_p2p_params_t * param,
    const uint data_size
){
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    if (param -> send_sz > 0){
        // if (threadIdx.x == 0){
        //     printf("-[RANK %u] waits for stage [%u] from RANK %u on credit[%u] to send %lu bytes\n", this_rank, stage_id, param -> dst_rank, credit_idx, param -> send_sz);
        // }
        wait_and_consume_credit(credit, credit_idx, stage_id, this_rank);

        // if (threadIdx.x == 0){
        //     printf("-[RANK %u] gets stage [%u] on credit[%u] to send %lu bytes\n", this_rank, stage_id, credit_idx, param -> send_sz );
        // }

        if (warp_id == 0){
            // use warp 0 in block 0 to do inter-node transfer
            // gpu i only talks to gpu i at another server
            nvshmemx_putmem_signal_nbi_warp(
                recv_buffer + param -> dst_disp * data_size,
                send_buffer + param -> src_disp * data_size,
                param -> send_sz * data_size,
                &sync_signal[this_rank],
                param -> send_sz,
                NVSHMEM_SIGNAL_ADD,
                param -> dst_rank
            );
            nvshmem_quiet();
            // if (lane_id == 0){
            //     nvshmem_uint64_wait_until(&sync_signal[param -> src_rank], NVSHMEM_CMP_EQ, param -> recv_sz);
            //     // sync_signal[param -> src_rank] = 0;
            // }
            // __syncwarp(); 
        }
    }
}

__global__ void flash_intranode_alltoall_redistribute(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    void ** remote_signal,
    const uint32_t remote_signal_offset,
    uint64_t * redis_complete_signal,
    // metadata for intranode transfer
    const struct device_intra_redistribute_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time,
    const uint data_size
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint dst_local_id = (local_rank_id + step) % local_rank_n;
        for (uint src_local_id = 0; src_local_id < local_rank_n; src_local_id ++){
            const uint64_t intra_round_n = __ldg(&param->round_n[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_offset = __ldg(&param->src_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            const uint64_t send_sz_this_rank = __ldg(&param->sz[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            const uint64_t recv_offset = __ldg(&param->dst_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            for (uint64_t round = 0; round < intra_round_n; round ++){
                const uint64_t tx_offset = round * min_tx_sz_at_a_time;
                const uint64_t send_offset_this_round = send_offset + tx_offset;
                const uint64_t recv_offset_this_round = recv_offset + tx_offset;
                uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);

                const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
                uint8_t * output_buf = (uint8_t *) remote_buffer[dst_local_id] + recv_offset_this_round;
                uint8_t * share_buf = (uint8_t *) shared_memory;

                // deal with misalignment
                uintptr_t inA  = (uintptr_t)input_buf;
                uintptr_t outA = (uintptr_t)output_buf;
                // assumption is that inA and outA has the same misalignment to enable int4-based data copy
                assert((inA & 0xF) == (outA & 0xF));

                uint64_t head = (16 - (inA & 0xF)) & 0xF;
                if (head > actual_send_sz_this_rank) head = actual_send_sz_this_rank;

                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        share_buf[t] = input_buf[t];
                    }
                }
                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        output_buf[t] = share_buf[t];
                    }
                }
                __syncthreads();
                input_buf  += head;
                output_buf += head;
                share_buf += sizeof(int4);
                actual_send_sz_this_rank -= head;

                const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
                const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
                const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
                const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;
                const uint64_t vec_bytes = total_vec_n * sizeof(int4);
                const uint64_t byte_tail = actual_send_sz_this_rank - vec_bytes;
                // STEP1: copy data from input buffer to shared buffer
                const int4* input_vec = (const int4*)input_buf;
                int4* share_vec = (int4*) share_buf;
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        share_buf[vec_bytes + t] = input_buf[vec_bytes + t];
                    }
                }
                // __syncthreads(); // Ensure shared memory write is complete

                // STEP2: copy data from shared buffer to output buffer
                int4* output_vec = (int4*)output_buf;
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        output_buf[vec_bytes + t] = share_buf[vec_bytes + t];
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete before output
            }
        }
        __threadfence_system();
        // notify the receiver that the transfer is completed
        if (threadIdx.x == 0){
            uint64_t * signal = ((uint64_t *) remote_signal[dst_local_id]) + remote_signal_offset + local_rank_id * block_n + block_id;
            atomicAdd64(signal, 1);
        }
        // __threadfence_system();
    }
    if (threadIdx.x == 1){
        atomicSet64(&redis_complete_signal[block_id], 1);
    }
    // __threadfence();    
}

__global__ void flash_intranode_alltoall_redistribute_no_signal(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    // void ** remote_signal,
    // const uint32_t remote_signal_offset,
    uint64_t * redis_complete_signal,
    // metadata for intranode transfer
    const struct device_intra_redistribute_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time,
    const uint data_size
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint dst_local_id = (local_rank_id + step) % local_rank_n;
        for (uint src_local_id = 0; src_local_id < local_rank_n; src_local_id ++){
            const uint64_t intra_round_n = __ldg(&param->round_n[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_offset = __ldg(&param->src_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            const uint64_t send_sz_this_rank = __ldg(&param->sz[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            const uint64_t recv_offset = __ldg(&param->dst_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]) * data_size;
            for (uint64_t round = 0; round < intra_round_n; round ++){
                const uint64_t tx_offset = round * min_tx_sz_at_a_time;
                const uint64_t send_offset_this_round = send_offset + tx_offset;
                const uint64_t recv_offset_this_round = recv_offset + tx_offset;
                uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);

                const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
                uint8_t * output_buf = (uint8_t *) remote_buffer[dst_local_id] + recv_offset_this_round;
                uint8_t * share_buf = (uint8_t *) shared_memory;

                // // deal with misalignment
                uintptr_t inA  = (uintptr_t)input_buf;
                uintptr_t outA = (uintptr_t)output_buf;
                // assumption is that inA and outA has the same misalignment to enable int4-based data copy
                assert((inA & 0xF) == (outA & 0xF));

                uint64_t head = (16 - (inA & 0xF)) & 0xF;
                if (head > actual_send_sz_this_rank) head = actual_send_sz_this_rank;

                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        share_buf[t] = input_buf[t];
                    }
                }
                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        output_buf[t] = share_buf[t];
                    }
                }
                __syncthreads();
                input_buf  += head;
                output_buf += head;
                share_buf += sizeof(int4);
                actual_send_sz_this_rank -= head;

                const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
                const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
                const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
                const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;
                const uint64_t vec_bytes = total_vec_n * sizeof(int4);
                const uint64_t byte_tail = actual_send_sz_this_rank - vec_bytes;
                // STEP1: copy data from input buffer to shared buffer
                const int4* input_vec = (const int4*)input_buf;
                int4* share_vec = (int4*) share_buf;
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        share_buf[vec_bytes + t] = input_buf[vec_bytes + t];
                    }
                }
                // __syncthreads(); // Ensure shared memory write is complete

                // STEP2: copy data from shared buffer to output buffer
                int4* output_vec = (int4*)output_buf;
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        output_buf[vec_bytes + t] = share_buf[vec_bytes + t];
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete before output
            }
        }
    }
    if (threadIdx.x == 0){
        atomicSet64(&redis_complete_signal[block_id], 1);
    }
}

__global__ void flash_intranode_alltoall_redistribute_simple(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    void ** remote_signal,
    const uint32_t remote_signal_offset,
    uint64_t * redis_complete_signal,
    // metadata for intranode transfer
    const struct device_intra_redistribute_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint dst_local_id = (local_rank_id + step) % local_rank_n;
        for (uint src_local_id = 0; src_local_id < local_rank_n; src_local_id ++){
            const uint64_t intra_round_n = __ldg(&param->round_n[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_offset = __ldg(&param->src_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_sz_this_rank = __ldg(&param->sz[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t recv_offset = __ldg(&param->dst_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            for (uint64_t round = 0; round < intra_round_n; round ++){
                const uint64_t tx_offset = round * min_tx_sz_at_a_time;
                const uint64_t send_offset_this_round = send_offset + tx_offset;
                const uint64_t recv_offset_this_round = recv_offset + tx_offset;
                const uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);

                const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
                uint8_t * output_buf = (uint8_t *) remote_buffer[dst_local_id] + recv_offset_this_round;
                uint8_t * share_buf = (uint8_t *) shared_memory;
        
                const uint64_t full_warp_n = actual_send_sz_this_rank / THREAD_N_PER_WARP;
                const uint64_t remainder_byte_start = full_warp_n * THREAD_N_PER_WARP;
                const uint64_t remainder_bytes = actual_send_sz_this_rank - remainder_byte_start;

                // STEP1: copy data from input buffer to shared buffer
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    share_buf[j * THREAD_N_PER_WARP + lane_id] = input_buf[j * THREAD_N_PER_WARP + lane_id];
                }
                if (warp_id == 0){
                    for (uint i = lane_id; i < remainder_bytes; i += THREAD_N_PER_WARP) {
                        share_buf[remainder_byte_start + i] = input_buf[remainder_byte_start + i];   // each thread does int4 transmission
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete

                // STEP2: copy data from shared buffer to output buffer

                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    output_buf[j * THREAD_N_PER_WARP + lane_id] = share_buf[j * THREAD_N_PER_WARP + lane_id];
                }
                if (warp_id == 0){
                    for (uint i = lane_id; i < remainder_bytes; i += THREAD_N_PER_WARP) {
                        output_buf[remainder_byte_start + i] = share_buf[remainder_byte_start + i];   // each thread does int4 transmission
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete before output
            }

        }
        cooperative_groups::this_grid().sync();
        __threadfence_system();
        // notify the receiver that the transfer is completed
        if (block_id == 0 && threadIdx.x == 0){
            uint64_t * signal = ((uint64_t *) remote_signal[dst_local_id]) + remote_signal_offset + local_rank_id;
            atomicAdd64(signal, 1);
        }
        if (block_id == 0 && threadIdx.x == 1){
            atomicSet64(redis_complete_signal, 1);
        }
    }
}

__global__ void flash_intranode_alltoall(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    void ** remote_signal,
    const uint32_t remote_signal_offset,
    // metadata for intranode transfer
    const struct device_intra_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time,
    const uint data_size
){

    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint peer_local_id = (local_rank_id + step) % local_rank_n;
        const uint64_t intra_round_n = __ldg(&param->round_n[peer_local_id * block_n + block_id]);
        const uint64_t send_offset = __ldg(&param->src_disp[peer_local_id * block_n + block_id]) * data_size;
        const uint64_t send_sz_this_rank = __ldg(&param->sz[peer_local_id * block_n + block_id]) * data_size;
        const uint64_t recv_offset = __ldg(&param->dst_disp[peer_local_id * block_n + block_id]) * data_size;
        for (uint64_t round = 0; round < intra_round_n; round ++){
            const uint64_t tx_offset = round * min_tx_sz_at_a_time;
            const uint64_t send_offset_this_round = send_offset + tx_offset;
            const uint64_t recv_offset_this_round = recv_offset + tx_offset;
            uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
           

            const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
            uint8_t * output_buf = (uint8_t *) remote_buffer[peer_local_id] + recv_offset_this_round;
            uint8_t * share_buf = (uint8_t *) shared_memory;

            // deal with misalignment
            uintptr_t inA  = (uintptr_t)input_buf;
            uintptr_t outA = (uintptr_t)output_buf;

            // assumption is that inA and outA has the same misalignment to enable int4-based data copy
            assert((inA & 0xF) == (outA & 0xF));

            uint64_t head = (16 - (inA & 0xF)) & 0xF;
            if (head > actual_send_sz_this_rank) head = actual_send_sz_this_rank;

            if (warp_id == 0) {
                for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                    share_buf[t] = input_buf[t];
                }
            }
            if (warp_id == 0) {
                for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                    output_buf[t] = share_buf[t];
                }
            }
            __syncthreads();
            input_buf  += head;
            output_buf += head;
            share_buf += sizeof(int4);
            actual_send_sz_this_rank -= head;

            const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
            const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
            const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
            const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;
            const uint64_t vec_bytes = total_vec_n * sizeof(int4);
            const uint64_t byte_tail = actual_send_sz_this_rank - vec_bytes;
           
            // STEP1: copy data from input buffer to shared buffer
            const int4* input_vec = (const int4*)input_buf;
            int4* share_vec = (int4*) share_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                }
            }
            if (warp_id == 0){
                for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                    share_buf[vec_bytes + t] = input_buf[vec_bytes + t];
                }
            }
            // __syncthreads(); // Ensure shared memory write is complete

            // STEP2: copy data from shared buffer to output buffer
            int4* output_vec = (int4*)output_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                }
           
            }
            if (warp_id == 0){
                for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                    output_buf[vec_bytes + t] = share_buf[vec_bytes + t];
                }
            }
            __syncthreads(); // Ensure shared memory write is complete before output
        }
        // synchronize all the blocks
        __threadfence_system();
        // notify the receiver that the transfer is completed
        if (threadIdx.x == 0){
            uint64_t * signal = ((uint64_t *) remote_signal[peer_local_id]) + remote_signal_offset + local_rank_id * block_n + block_id;
            atomicSet64(signal, 1);
        }
        // __threadfence_system();
    }
}

__global__ void flash_intranode_alltoall_no_signal(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    // metadata for intranode transfer
    const struct device_intra_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time,
    const uint data_size
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint peer_local_id = (local_rank_id + step) % local_rank_n;
        const uint64_t intra_round_n = __ldg(&param->round_n[peer_local_id * block_n + block_id]);
        const uint64_t send_offset = __ldg(&param->src_disp[peer_local_id * block_n + block_id]) * data_size;
        const uint64_t send_sz_this_rank = __ldg(&param->sz[peer_local_id * block_n + block_id]) * data_size;
        const uint64_t recv_offset = __ldg(&param->dst_disp[peer_local_id * block_n + block_id]) * data_size;
        for (uint64_t round = 0; round < intra_round_n; round ++){
            const uint64_t tx_offset = round * min_tx_sz_at_a_time;
            const uint64_t send_offset_this_round = send_offset + tx_offset;
            const uint64_t recv_offset_this_round = recv_offset + tx_offset;
            uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
           

            const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
            uint8_t * output_buf = (uint8_t *) remote_buffer[peer_local_id] + recv_offset_this_round;
            uint8_t * share_buf = (uint8_t *) shared_memory;

            // // deal with misalignment
            uintptr_t inA  = (uintptr_t)input_buf;
            uintptr_t outA = (uintptr_t)output_buf;

            // assumption is that inA and outA has the same misalignment to enable int4-based data copy
            assert((inA & 0xF) == (outA & 0xF));

            uint64_t head = (16 - (inA & 0xF)) & 0xF;
            if (head > actual_send_sz_this_rank) head = actual_send_sz_this_rank;

            if (warp_id == 0) {
                for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                    share_buf[t] = input_buf[t];
                }
            }
            if (warp_id == 0) {
                for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                    output_buf[t] = share_buf[t];
                }
            }
            __syncthreads();
            input_buf  += head;
            output_buf += head;
            share_buf += sizeof(int4);
            actual_send_sz_this_rank -= head;

            const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
            const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
            const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
            const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;
            const uint64_t vec_bytes = total_vec_n * sizeof(int4);
            const uint64_t byte_tail = actual_send_sz_this_rank - vec_bytes;
           
            // STEP1: copy data from input buffer to shared buffer
            const int4* input_vec = (const int4*)input_buf;
            int4* share_vec = (int4*) share_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                }
            }
            if (warp_id == 0){
                for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                    share_buf[vec_bytes + t] = input_buf[vec_bytes + t];
                }
            }
            // __syncthreads(); // Ensure shared memory write is complete

            // STEP2: copy data from shared buffer to output buffer
            int4* output_vec = (int4*)output_buf;
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                }
                #pragma unroll
                for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                    output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                }
            }
            if (warp_id == warp_n - 1){
                for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                    output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                }
           
            }
            if (warp_id == 0){
                for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                    output_buf[vec_bytes + t] = share_buf[vec_bytes + t];
                }
            }
            __syncthreads(); // Ensure shared memory write is complete before output
        }
        // __threadfence_system();
    }
    // __threadfence_system();
}

__global__ void flash_intranode_alltoall_simple(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    void ** remote_signal,
    const uint32_t remote_signal_offset,
    // metadata for intranode transfer
    const struct device_intra_alltoall_params_t * param,
    const uint64_t  min_tx_sz_at_a_time
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint peer_local_id = (local_rank_id + step) % local_rank_n;
        const uint64_t intra_round_n = __ldg(&param->round_n[peer_local_id * block_n + block_id]);
        const uint64_t send_offset = __ldg(&param->src_disp[peer_local_id * block_n + block_id]);
        const uint64_t send_sz_this_rank = __ldg(&param->sz[peer_local_id * block_n + block_id]);
        const uint64_t recv_offset = __ldg(&param->dst_disp[peer_local_id * block_n + block_id]);
        for (uint64_t round = 0; round < intra_round_n; round ++){
            const uint64_t tx_offset = round * min_tx_sz_at_a_time;
            const uint64_t send_offset_this_round = send_offset + tx_offset;
            const uint64_t recv_offset_this_round = recv_offset + tx_offset;
            uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
           

            const uint8_t * input_buf = (uint8_t *) buffer_in + send_offset_this_round;
            uint8_t * output_buf = (uint8_t *) remote_buffer[peer_local_id] + recv_offset_this_round;
            uint8_t * share_buf = (uint8_t *) shared_memory;
           
            // STEP1: copy data from input buffer to shared buffer
            const uint64_t full_warp_n = actual_send_sz_this_rank / THREAD_N_PER_WARP;
            const uint64_t remainder_byte_start = full_warp_n * THREAD_N_PER_WARP;
            const uint64_t remainder_bytes = actual_send_sz_this_rank - remainder_byte_start;

            // STEP1: copy data from input buffer to shared buffer
            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                share_buf[j * THREAD_N_PER_WARP + lane_id] = input_buf[j * THREAD_N_PER_WARP + lane_id];
            }
            if (warp_id == 0){
                for (uint i = lane_id; i < remainder_bytes; i += THREAD_N_PER_WARP) {
                    share_buf[remainder_byte_start + i] = input_buf[remainder_byte_start + i];   // each thread does int4 transmission
                }
            }
            __syncthreads(); // Ensure shared memory write is complete

            // STEP2: copy data from shared buffer to output buffer

            for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                output_buf[j * THREAD_N_PER_WARP + lane_id] = share_buf[j * THREAD_N_PER_WARP + lane_id];
            }
            if (warp_id == 0){
                for (uint i = lane_id; i < remainder_bytes; i += THREAD_N_PER_WARP) {
                    output_buf[remainder_byte_start + i] = share_buf[remainder_byte_start + i];   // each thread does int4 transmission
                }
            }
            __syncthreads(); // Ensure shared memory write is complete before output
        }
        cooperative_groups::this_grid().sync();
        __threadfence_system();
        // notify the receiver that the transfer is completed
        if (block_id == 0 && threadIdx.x == 0){
            uint64_t * signal = ((uint64_t *) remote_signal[peer_local_id]) + remote_signal_offset + local_rank_id;
            atomicSet64(signal, 1);
        }
    }
}

__global__ void wait_stage_ready(uint64_t * sync_signal, struct device_inter_p2p_params_t * p, uint stage_id, uint64_t threshold){
  if (threadIdx.x == 0){
    // printf("[NODE %u] wait from src rank %u for stage %u, current val: %lu, threshold: %lu\n", p -> dst_rank,  p->src_rank, stage_id, sync_signal[p -> src_rank], threshold);
    nvshmem_uint64_wait_until(&sync_signal[p -> src_rank], NVSHMEM_CMP_GE, p -> recv_sz);
    sync_signal[p -> src_rank] -= p -> recv_sz;
  }
}

__global__ void wait_balance_complete(uint64_t * sync_signal, const uint32_t local_rank_n, const uint32_t block_n, const struct device_intra_alltoall_params_t * p){
    for (uint i = threadIdx.x; i < local_rank_n * block_n; i += blockDim.x){
        while (true){
            uint64_t val = *((volatile uint64_t*)(sync_signal + i));
            if (val == 1) {
                sync_signal[i] = 0;
                // atomicSet64(&sync_signal[i], 0);
                break;
            }
        }
    }
}

__global__ void wait_redistribute_complete(uint64_t * sync_signal, const uint32_t local_rank_n, const int this_rank, const uint block_n){
    for (uint i = threadIdx.x; i < local_rank_n * block_n; i += blockDim.x){
        while (true){
            uint64_t val = *((volatile uint64_t*)(sync_signal + i));
            if (val == 1) {
                sync_signal[i] = 0; //reset signal 
                break;
            }
        }
    }
}

int launch_flash_alltoallv(
    struct flash_buffer_ptr_t * buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t stream3,
    cudaEvent_t * events,
    uint event_n,
    uint data_size,
    uint credit_flip    //0 or 1
){
    uint32_t block_n = k_sz -> block_n;
    uint32_t this_rank = flash_sched->info.this_rank;
    uint32_t local_rank_n = flash_sched->info.local_rank_n;
    uint32_t rank_n = flash_sched -> info.rank_n;
    uint8_t * cur_recvbuff = buf->internode_buffer1, * prev_recvbuff = buf->internode_buffer2;
    uint64_t * cur_sync = buf->sync_signal1, * prev_sync = buf->sync_signal2, * cur_redis_signal, * prev_redis_signal;
    struct device_inter_p2p_params_t * cur_p2p, * prev_p2p;
    struct device_intra_redistribute_alltoall_params_t * cur_redistribute;
    uint stage_signal_id = 0, next_stage_src_rank = 0, cur_credit_idx = 0, next_credit_idx = 1, recv_signal_offset = 0;

    // reset counters
    // CUDACHECK(cudaMemsetAsync(buf->recv_complete_signal, 0, sizeof(uint64_t) * local_rank_n * (flash_sched -> stage_n + 1), stream1));
    // CUDACHECK(cudaMemsetAsync(buf->sync_signal1, 0, sizeof(uint64_t) * rank_n, stream1));
    // CUDACHECK(cudaMemsetAsync(buf->sync_signal2, 0, sizeof(uint64_t) * rank_n, stream1));
    uint next_round_offset = 2 * rank_n * (credit_flip % 2);
    uint this_round_offset = 2 * rank_n * ((credit_flip + 1) % 2);
    uint64_t * credit_buffer_this_round = &(buf->credit[this_round_offset]);
    CUDACHECK(cudaMemsetAsync(&buf->credit[next_round_offset], 0, sizeof(uint64_t) * 2 * rank_n, stream1));
    grant_initial_credit_to_peer<<<1, 1, 0, stream1>>>(credit_buffer_this_round, this_rank, flash_sched -> host_stages_internode[0].src_rank, 1);
    CUCHECK(cuEventRecord(events[1], stream1));
    CUDACHECK(cudaStreamWaitEvent(stream3, events[1], 0));  // following credit grant must be after the initial grant

    // use stream2 for intra-node communication
    // balance alltoall
    void * balance_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.balance_recv_buffer.remote_data,
        &buf->balance_send_buffer,
        &buf->share_mem.recv_complete_signal.remote_data,
        &recv_signal_offset,
        &flash_sched->dev_balance_alltoall,
        &k_sz->min_tx_sz_at_a_time,
        &data_size
    };
    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        balance_alltoall_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    // // copy data from balance recv buffer to send buffer
    uint64_t * recv_signal = ((uint64_t *) buf->recv_complete_signal) + recv_signal_offset;
    wait_balance_complete<<<1, local_rank_n * block_n, 0, stream2>>>(recv_signal, local_rank_n, block_n, flash_sched->dev_balance_alltoall);
    for (uint i = 0; i < flash_sched -> balance_memcpy_n; i ++){
        CUDACHECK(cudaMemcpyAsync(((char*)buf->send_buffer + flash_sched -> balance_memcpy[i].dst_disp * data_size),
                         ((char*) buf->balance_recv_buffer + flash_sched -> balance_memcpy[i].src_disp * data_size),
                          flash_sched -> balance_memcpy[i].sz * data_size,
                          cudaMemcpyDeviceToDevice,
                          stream2));
    }
    CUCHECK(cuEventRecord(events[0], stream2));

    // start internode transfer
    // first pipeline stage
    CUDACHECK(cudaStreamWaitEvent(stream1, events[0], 0));
    cur_p2p = flash_sched -> dev_stages_internode;
    cur_credit_idx = flash_sched -> host_stages_internode[0].dst_rank;
    stage_signal_id = 1;    // need to receive credit before run to avoid cross-iteration test issue
    void * internode_first_stage[] = {
        &this_rank,
        &buf->send_buffer,
        &cur_recvbuff,
        &cur_sync,
        &credit_buffer_this_round,
        &cur_credit_idx,
        &stage_signal_id,
        &cur_p2p,
        &data_size
    };
    NVSHMEMCHECK(cudaLaunchKernel(
        (void *) &flash_internode_p2p,
        dim3(1), 
        dim3(32),
        internode_first_stage,
        0,
        stream1
    ));

    // intrinsic alltoall - after load balance
    void * intrinsic_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.out_buffer.remote_data,
        &buf->buffer_in,
        &flash_sched->dev_intrinsic_alltoall,
        &k_sz -> min_tx_sz_at_a_time,
        &data_size
    };

    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall_no_signal,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        intrinsic_alltoall_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    // middle pipeline stages
    for (uint step_id = 1; step_id < flash_sched -> stage_n - 1; step_id ++){
    // for (uint step_id = 1; step_id < 1; step_id ++){
        prev_recvbuff = (step_id % 2 == 1) ? buf->internode_buffer1 : buf->internode_buffer2;
        cur_recvbuff = (step_id % 2 == 1) ? buf->internode_buffer2 : buf->internode_buffer1;
        cur_sync = (step_id % 2 == 1) ? buf->sync_signal2 : buf->sync_signal1;
        prev_sync = (step_id % 2 == 1) ? buf->sync_signal1 : buf->sync_signal2;
        prev_p2p = &(flash_sched -> dev_stages_internode[step_id - 1]);
        cur_p2p = &(flash_sched -> dev_stages_internode[step_id]);
        cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[step_id]);

        if (flash_sched -> host_stages_internode[step_id].recv_sz> 0){
            next_stage_src_rank = flash_sched -> host_stages_internode[step_id].src_rank;
            next_credit_idx = this_rank + (step_id % 2) * rank_n;
            cur_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[step_id * block_n];
            grant_credit_to_peer<<<1, block_n, 0, stream3>>>(prev_sync, cur_redis_signal, prev_p2p, credit_buffer_this_round, next_credit_idx, next_stage_src_rank, step_id, this_rank, block_n, flash_sched -> host_sync_threshold[step_id - 1]);
        }
        wait_stage_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, step_id, flash_sched -> host_sync_threshold[step_id - 1]);
        
        cur_credit_idx = flash_sched -> host_stages_internode[step_id].dst_rank + (step_id % 2) * rank_n;
        stage_signal_id = step_id;

        void * internode_cur_stage[] = {
            &this_rank,
            &buf->send_buffer,
            &cur_recvbuff,
            &cur_sync,
            &credit_buffer_this_round,
            &cur_credit_idx,
            &stage_signal_id,
            &cur_p2p,
            &data_size
        };
        NVSHMEMCHECK(cudaLaunchKernel(
            (void *) &flash_internode_p2p,
            dim3(1), 
            dim3(32),
            internode_cur_stage,
            0,
            stream1
        ));        
        
        // recv_signal_offset += local_rank_n * block_n;
        prev_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[(step_id + 1) * block_n];
        void * stage_redistribute_args[] = {
            &this_rank,
            &local_rank_n,
            &buf->share_mem.out_buffer.remote_data,
            &prev_recvbuff,
            &prev_redis_signal,
            &cur_redistribute,
            &k_sz -> min_tx_sz_at_a_time,
            &data_size
        };

        CUDACHECK(cudaLaunchKernel(
        (void *)&flash_intranode_alltoall_redistribute_no_signal,
            k_sz -> grid_dim,
            k_sz -> block_dim, 
            stage_redistribute_args,
            k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
            stream2
        ));  
    }

    // final pipeline stage - final data redistribution
    prev_sync = ((flash_sched -> stage_n - 1) % 2 == 1) ? buf->sync_signal1 : buf->sync_signal2;
    prev_recvbuff = ((flash_sched -> stage_n - 1) % 2 == 1) ? buf->internode_buffer1 : buf->internode_buffer2;
    prev_p2p = &(flash_sched -> dev_stages_internode[flash_sched -> stage_n - 2]);
    cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[flash_sched -> stage_n - 1]);
    prev_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[0];

    void * final_redistribute_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.out_buffer.remote_data,
        &prev_recvbuff,
        &prev_redis_signal,
        &cur_redistribute,
        &k_sz -> min_tx_sz_at_a_time,
        &data_size
    };

    wait_stage_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, flash_sched -> stage_n - 1, flash_sched -> host_sync_threshold[flash_sched -> stage_n - 2]);

    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall_redistribute_no_signal,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        final_redistribute_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    return NVSHMEMX_SUCCESS;
}


__global__ void flash_internode_p2p_chunk(
    uint32_t this_rank,
    // nvshmem memory for RDMA data exchange
    uint8_t* send_buffer,
    uint8_t* recv_buffer,
    uint64_t* sync_signal,      // symmetric, per-src slot
    // credit
    uint64_t * credit,
    const uint32_t credit_idx,
    const uint32_t stage_id,
    const device_inter_p2p_params_t * p, // has src_disp, dst_disp, send_sz, dst_rank, src_rank
    const chunk_metadata_t * c
){
    const uint32_t warp_id = threadIdx.x / 32;
    if (p -> send_sz > 0){
        wait_and_consume_credit(credit, credit_idx, stage_id, this_rank);
        
        if (warp_id == 0) {
            for (uint chunkid = 0; chunkid < c -> chunk_n; chunkid++){
                nvshmemx_putmem_signal_nbi_warp(
                    recv_buffer + p->dst_disp + c->chunk_disp[chunkid],
                    send_buffer + p->src_disp + c->chunk_disp[chunkid],
                    c->chunk_sz[chunkid],
                    &sync_signal[this_rank],
                    c->chunk_sz[chunkid],
                    NVSHMEM_SIGNAL_ADD,      // or use ADD with n and wait >=
                    p->dst_rank);
            }
            nvshmem_quiet();  // optional: only if you need to bound completion here.
        }
    }
}

__global__ void flash_intranode_alltoall_redistribute_chunk(
    // rank information
    const uint32_t this_rank,
    const uint32_t local_rank_n,
    // memory for IPC memory copy using NVLINK
    void ** remote_buffer,
    const void * buffer_in,
    // metadata for intranode transfer
    uint64_t * redis_complete_signal,
    const struct device_intra_redistribute_alltoall_params_t * param,
    const chunk_metadata_t * c,
    const uint32_t chunk_id,
    const uint64_t  min_tx_sz_at_a_time
){
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_n = gridDim.x;
    const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
    const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
    const uint32_t warp_n = blockDim.x / THREAD_N_PER_WARP;
    
    const uint32_t local_rank_id = this_rank % local_rank_n;
    
    // use shared memory within thread block cluster to improve performance
    extern __shared__ uint8_t shared_memory[];  // size: min_tx_sz_at_a_time

    for (uint step = 0; step < local_rank_n; step ++){
        const uint dst_local_id = (local_rank_id + step) % local_rank_n;
        for (uint src_local_id = 0; src_local_id < local_rank_n; src_local_id ++){
            const uint64_t intra_round_n = __ldg(&param->round_n[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_offset = __ldg(&param->src_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t send_sz_this_rank = __ldg(&param->sz[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            const uint64_t recv_offset = __ldg(&param->dst_disp[(dst_local_id * local_rank_n + src_local_id) * block_n + block_id]);
            for (uint64_t round = 0; round < intra_round_n; round ++){
                const uint64_t tx_offset = round * min_tx_sz_at_a_time;
                // This round send ranges: [send_offset_this_round, send_offset_this_round + actual_send_sz_this_rank)
                const uint64_t send_offset_this_round = send_offset + tx_offset;
                uint64_t actual_send_sz_this_rank = MIN(min_tx_sz_at_a_time, send_sz_this_rank - tx_offset);
                // current chunk range: [param -> chunk_disp[chunk_id], param -> chunk_disp[chunk_id] + param -> chunk_sz[chunk_id])
                if (send_offset_this_round + actual_send_sz_this_rank <= c -> chunk_disp[chunk_id] || send_offset_this_round >= c -> chunk_disp[chunk_id] + c -> chunk_sz[chunk_id]) continue;

                // Compute overlap region
                const uint64_t overlap_disp = MAX(send_offset_this_round, c -> chunk_disp[chunk_id]);
                const uint64_t overlap_sz = MIN(send_offset_this_round + actual_send_sz_this_rank, c -> chunk_disp[chunk_id] + c -> chunk_sz[chunk_id]) - overlap_disp;
                const uint64_t recv_offset_this_round = recv_offset + tx_offset;

                // deal with misalignment first
                const uint8_t * input_buf = (uint8_t *) buffer_in + overlap_disp;
                uint8_t * output_buf = (uint8_t *) remote_buffer[dst_local_id] + recv_offset_this_round + overlap_disp - send_offset_this_round;
                uint8_t * share_buf = (uint8_t *) shared_memory;

                uintptr_t inA  = (uintptr_t)input_buf;
                uintptr_t outA = (uintptr_t)output_buf;

                // assumption is that inA and outA has the same misalignment to enable int4-based data copy
                assert((inA & 0xF) == (outA & 0xF));

                uint64_t head = (16 - (inA & 0xF)) & 0xF;
                if (head > actual_send_sz_this_rank) head = actual_send_sz_this_rank;

                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        share_buf[t] = input_buf[t];
                    }
                }
                if (warp_id == 0) {
                    for (uint64_t t = lane_id; t < head; t += THREAD_N_PER_WARP) {
                        output_buf[t] = share_buf[t];
                    }
                }
                __syncthreads();
                input_buf  += head;
                output_buf += head;
                share_buf += sizeof(int4);
                actual_send_sz_this_rank -= head;

                const uint64_t total_vec_n = actual_send_sz_this_rank / sizeof(int4);
                const uint64_t full_warp_n = total_vec_n / (TX_UNROLL_FACTOR * THREAD_N_PER_WARP); // each warp sends TX_UNROLL_FACTOR * sizeof(int4) * THREAD_N_PER_WARP, each thread sends  4 * sizeof(int4)
                const uint64_t vec_tail_start = full_warp_n * TX_UNROLL_FACTOR * THREAD_N_PER_WARP;
                const uint64_t remainder_vec_n = total_vec_n - vec_tail_start;
                const uint64_t vec_bytes = total_vec_n * sizeof(int4);
                const uint64_t byte_tail = actual_send_sz_this_rank - vec_bytes;
                           
                // STEP1: copy data from input buffer to shared buffer
                const int4* input_vec = (const int4*)input_buf;
                int4* share_vec = (int4*) share_buf;
                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 input_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        input_reg[k] = input_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        share_vec[thread_offset + k * THREAD_N_PER_WARP] = input_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        share_vec[vec_tail_start + i] = input_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        share_buf[vec_bytes + t] = input_buf[vec_bytes + t];
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete

                // STEP2: copy data from shared buffer to output buffer
                int4* output_vec = (int4*)output_buf;

                for(uint64_t j = warp_id; j < full_warp_n; j += warp_n){ 
                    const uint64_t thread_offset = j * (TX_UNROLL_FACTOR * THREAD_N_PER_WARP) + lane_id;
                    int4 output_reg[TX_UNROLL_FACTOR]; // local register at each thread for staged transfer
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_reg[k] = share_vec[thread_offset + k * THREAD_N_PER_WARP];
                    }
                    #pragma unroll
                    for (uint k = 0; k < TX_UNROLL_FACTOR; k ++){
                        output_vec[thread_offset + k * THREAD_N_PER_WARP] = output_reg[k];
                    }
                }
                if (warp_id == warp_n - 1){
                    for (uint i = lane_id; i < remainder_vec_n; i += THREAD_N_PER_WARP) {
                        output_vec[vec_tail_start + i] = share_vec[vec_tail_start + i];   // each thread does int4 transmission
                    }
                }
                if (warp_id == 0){
                    for (uint64_t t = lane_id; t < byte_tail; t += THREAD_N_PER_WARP) {
                        output_buf[vec_bytes + t] = share_buf[vec_bytes + t];
                    }
                }
                __syncthreads(); // Ensure shared memory write is complete before output
            }
        }
    }
    if (threadIdx.x == 0){
        if (chunk_id == c -> chunk_n - 1){
            atomicSet64(&redis_complete_signal[block_id], 1);
        }
    }
}

__global__ void signal_redis_completion(uint64_t * redis_complete_signal, const uint32_t block_n){
    for (uint i = threadIdx.x; i < block_n; i += blockDim.x){
        atomicSet64(&redis_complete_signal[i], 1);
    }
}



__global__ void wait_chunk_ready(uint64_t * sync_signal, const struct device_inter_p2p_params_t * inter, const struct chunk_metadata_t * c, const uint32_t chunkid){
  if (threadIdx.x == 0){
    nvshmem_uint64_wait_until(&sync_signal[inter -> src_rank], NVSHMEM_CMP_GE, c -> chunk_ready_threshold[chunkid]);
    // printf("WATI CHUNK AFTER: RANK %u => RANK %u - chunk_id: %u\n", inter -> src_rank, inter -> dst_rank, chunkid);
    if(c -> chunk_ready_threshold[chunkid] == inter->recv_sz){
        sync_signal[inter -> src_rank] -= inter->recv_sz;
        // st_uint64_volatile(&sync_signal[inter -> src_rank], 0);
    }
  }
}


int launch_flash_alltoallv_chunk(
    struct flash_buffer_ptr_t * buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t stream3,
    cudaEvent_t * events,
    uint event_n,
    uint credit_flip    //0 or 1
){
    uint32_t block_n = k_sz -> block_n;
    uint32_t this_rank = flash_sched->info.this_rank;
    uint32_t local_rank_n = flash_sched->info.local_rank_n;
    uint32_t rank_n = flash_sched -> info.rank_n;
    uint8_t * cur_recvbuff = buf->internode_buffer1, * prev_recvbuff = buf->internode_buffer2;
    uint64_t * cur_sync = buf->sync_signal1, * prev_sync = buf->sync_signal2, * cur_redis_signal, * prev_redis_signal;
    struct device_inter_p2p_params_t * cur_p2p, * prev_p2p;
    struct device_intra_redistribute_alltoall_params_t * cur_redistribute;
    uint stage_signal_id = 0, next_stage_src_rank = 0, cur_credit_idx = 0, next_credit_idx = 1, recv_signal_offset = 0;
    struct chunk_metadata_t * send_chunk_param = flash_sched -> dev_stages_send_chunks, * recv_chunk_param;


    // reset counters
    uint next_round_offset = 2 * rank_n * (credit_flip % 2);
    uint this_round_offset = 2 * rank_n * ((credit_flip + 1) % 2);
    uint64_t * credit_buffer_this_round = &(buf->credit[this_round_offset]);

    CUDACHECK(cudaMemsetAsync(&buf->credit[next_round_offset], 0, sizeof(uint64_t) * 2 * rank_n, stream1));
    grant_initial_credit_to_peer<<<1, 1, 0, stream1>>>(credit_buffer_this_round, this_rank, flash_sched -> host_stages_internode[0].src_rank, 1);
    CUCHECK(cuEventRecord(events[1], stream1));
    CUDACHECK(cudaStreamWaitEvent(stream3, events[1], 0));  // following credit grant must be after the initial grant

    // use stream2 for intra-node communication
    // balance alltoall
    void * balance_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.balance_recv_buffer.remote_data,
        &buf->balance_send_buffer,
        &buf->share_mem.recv_complete_signal.remote_data,
        &recv_signal_offset,
        &flash_sched->dev_balance_alltoall,
        &k_sz->min_tx_sz_at_a_time,
    };
    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        balance_alltoall_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    // // copy data from balance recv buffer to send buffer
    uint64_t * recv_signal = ((uint64_t *) buf->recv_complete_signal) + recv_signal_offset;
    wait_balance_complete<<<1, local_rank_n * block_n, 0, stream2>>>(recv_signal, local_rank_n, block_n, flash_sched->dev_balance_alltoall);
    for (uint i = 0; i < flash_sched -> balance_memcpy_n; i ++){
        CUDACHECK(cudaMemcpyAsync(((char*)buf->send_buffer + flash_sched -> balance_memcpy[i].dst_disp),
                         ((char*) buf->balance_recv_buffer + flash_sched -> balance_memcpy[i].src_disp),
                          flash_sched -> balance_memcpy[i].sz,
                          cudaMemcpyDeviceToDevice,
                          stream2));
    }
    CUCHECK(cuEventRecord(events[0], stream2));

    // start internode transfer
    // first pipeline stage
    CUDACHECK(cudaStreamWaitEvent(stream1, events[0], 0));
    cur_p2p = flash_sched -> dev_stages_internode;
    cur_credit_idx = flash_sched -> host_stages_internode[0].dst_rank;
    stage_signal_id = 1;    // need to receive credit before run to avoid cross-iteration test issue
    void * internode_first_stage[] = {
        &this_rank,
        &buf->send_buffer,
        &cur_recvbuff,
        &cur_sync,
        &credit_buffer_this_round,
        &cur_credit_idx,
        &stage_signal_id,
        &cur_p2p,
        &send_chunk_param
    };

    NVSHMEMCHECK(cudaLaunchKernel(
        (void *) &flash_internode_p2p_chunk,
        dim3(1), 
        dim3(32),
        internode_first_stage,
        0,
        stream1
    ));

    // intrinsic alltoall - after load balance
    void * intrinsic_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.out_buffer.remote_data,
        &buf->buffer_in,
        &flash_sched->dev_intrinsic_alltoall,
        &k_sz -> min_tx_sz_at_a_time,
    };

    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall_no_signal,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        intrinsic_alltoall_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    // middle pipeline stages
    for (uint step_id = 1; step_id < flash_sched -> stage_n - 1; step_id ++){
    // for (uint step_id = 1; step_id < 1; step_id ++){
        prev_recvbuff = (step_id % 2 == 1) ? buf->internode_buffer1 : buf->internode_buffer2;
        cur_recvbuff = (step_id % 2 == 1) ? buf->internode_buffer2 : buf->internode_buffer1;
        cur_sync = (step_id % 2 == 1) ? buf->sync_signal2 : buf->sync_signal1;
        prev_sync = (step_id % 2 == 1) ? buf->sync_signal1 : buf->sync_signal2;
        prev_p2p = &(flash_sched -> dev_stages_internode[step_id - 1]);
        cur_p2p = &(flash_sched -> dev_stages_internode[step_id]);
        cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[step_id]);

        if (flash_sched -> host_stages_internode[step_id].recv_sz> 0){
            next_stage_src_rank = flash_sched -> host_stages_internode[step_id].src_rank;
            next_credit_idx = this_rank + (step_id % 2) * rank_n;
            cur_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[step_id * block_n];
            grant_credit_to_peer<<<1, block_n, 0, stream3>>>(prev_sync, cur_redis_signal, prev_p2p, credit_buffer_this_round, next_credit_idx, next_stage_src_rank, step_id, this_rank, block_n, flash_sched -> host_sync_threshold[step_id - 1]);
        }
        // wait_stage_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, step_id, flash_sched -> host_sync_threshold[step_id - 1]);
        
        cur_credit_idx = flash_sched -> host_stages_internode[step_id].dst_rank + (step_id % 2) * rank_n;
        stage_signal_id = step_id;

        void * internode_cur_stage[] = {
            &this_rank,
            &buf->send_buffer,
            &cur_recvbuff,
            &cur_sync,
            &credit_buffer_this_round,
            &cur_credit_idx,
            &stage_signal_id,
            &cur_p2p
        };
        NVSHMEMCHECK(cudaLaunchKernel(
            (void *) &flash_internode_p2p,
            dim3(1), 
            dim3(32),
            internode_cur_stage,
            0,
            stream1
        ));        
        
        // chunked redistribution
        recv_chunk_param = &(flash_sched -> dev_stages_recv_chunks[step_id]);
        const uint32_t chunk_n = flash_sched->host_stages_recv_chunks[step_id].chunk_n;
        prev_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[(step_id + 1) * block_n];
        for (uint chunkid = 0; chunkid < chunk_n; chunkid++){
            wait_chunk_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, recv_chunk_param, chunkid);
            void * stage_redistribute_args[] = {
                &this_rank,
                &local_rank_n,
                &buf->share_mem.out_buffer.remote_data,
                &prev_recvbuff,
                &prev_redis_signal,
                &cur_redistribute,
                &recv_chunk_param,
                &chunkid,
                &k_sz -> min_tx_sz_at_a_time
            };
            CUDACHECK(cudaLaunchKernel(
                (void *)&flash_intranode_alltoall_redistribute_chunk,
                    k_sz -> grid_dim,
                    k_sz -> block_dim, 
                    stage_redistribute_args,
                    k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
                    stream2
            ));
        }
        if (chunk_n == 0){
            signal_redis_completion<<<1, block_n, 0, stream2>>>(prev_redis_signal, block_n);
        } 
    }

    // final pipeline stage - final data redistribution
    prev_sync = ((flash_sched -> stage_n - 1) % 2 == 1) ? buf->sync_signal1 : buf->sync_signal2;
    prev_recvbuff = ((flash_sched -> stage_n - 1) % 2 == 1) ? buf->internode_buffer1 : buf->internode_buffer2;
    prev_p2p = &(flash_sched -> dev_stages_internode[flash_sched -> stage_n - 2]);
    cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[flash_sched -> stage_n - 1]);
    prev_redis_signal = &((uint64_t *)buf -> redistribute_complete_signal)[0];

    void * final_redistribute_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.out_buffer.remote_data,
        &prev_recvbuff,
        &prev_redis_signal,
        &cur_redistribute,
        &k_sz -> min_tx_sz_at_a_time
    };

    wait_stage_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, flash_sched -> stage_n - 1, flash_sched -> host_sync_threshold[flash_sched -> stage_n - 2]);

    CUDACHECK(cudaLaunchKernel(
    (void *)&flash_intranode_alltoall_redistribute_no_signal,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        final_redistribute_args,
        k_sz ->min_tx_sz_at_a_time + 16,      // shared memory size per block
        stream2
    ));

    CUDACHECK(cudaStreamSynchronize(stream2));
    CUDACHECK(cudaStreamSynchronize(stream1));
    return NVSHMEMX_SUCCESS;
}


int launch_flash_alltoallv_chunk_old(
    struct flash_buffer_ptr_t * buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1,
    cudaStream_t stream2
){
    uint32_t this_rank = flash_sched->info.this_rank;
    uint32_t local_rank_n = flash_sched->info.local_rank_n;
    uint32_t rank_n = flash_sched->info.rank_n;
    uint8_t * cur_recvbuff = buf->internode_buffer1, * prev_recvbuff = buf->internode_buffer2;
    uint64_t * cur_sync = buf->sync_signal1, * prev_sync = buf->sync_signal2;
    struct device_inter_p2p_params_t * cur_p2p, * prev_p2p;
    struct device_intra_redistribute_alltoall_params_t * cur_redistribute;
    struct chunk_metadata_t * send_chunk_param = flash_sched -> dev_stages_send_chunks, * recv_chunk_param;

    // use stream2 for intra-node communication
    // balance alltoall
    void * balance_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.balance_recv_buffer.remote_data,
        &buf->balance_send_buffer,
        &flash_sched->dev_balance_alltoall,
        &k_sz->min_tx_sz_at_a_time
    };
    CUDACHECK(cudaLaunchCooperativeKernel(
    (void *)&flash_intranode_alltoall,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        balance_alltoall_args,
        k_sz ->min_tx_sz_at_a_time,      // shared memory size per block
        stream1
    ));
    // copy data from balance recv buffer to send buffer
    for (uint i = 0; i < flash_sched -> balance_memcpy_n; i ++){
        CUDACHECK(cudaMemcpyAsync(((char*)buf->send_buffer + flash_sched -> balance_memcpy[i].dst_disp),
                         ((char*) buf->balance_recv_buffer + flash_sched -> balance_memcpy[i].src_disp),
                          flash_sched -> balance_memcpy[i].sz,
                          cudaMemcpyDeviceToDevice,
                          stream1));
    }
    // start internode transfer
    // first pipeline stage
    cur_p2p = flash_sched -> dev_stages_internode;
    void * internode_first_stage[] = {
        &this_rank,
        &buf->send_buffer,
        &cur_recvbuff,
        &cur_sync,
        &cur_p2p,
        &send_chunk_param
    };
    NVSHMEMCHECK(nvshmemx_collective_launch(
        (void *) &flash_internode_p2p_chunk,
        dim3(1), 
        dim3(32),
        internode_first_stage,
        0,
        stream1
    ));

    // intrinsic alltoall
    void * intrinsic_alltoall_args[] = {
        &this_rank,
        &local_rank_n,
        &buf->share_mem.out_buffer.remote_data,
        &buf->buffer_in,
        &flash_sched->dev_intrinsic_alltoall,
        &k_sz -> min_tx_sz_at_a_time
    };

    CUDACHECK(cudaLaunchCooperativeKernel(
    (void *)&flash_intranode_alltoall,
        k_sz -> grid_dim,
        k_sz -> block_dim, 
        intrinsic_alltoall_args,
        k_sz ->min_tx_sz_at_a_time,      // shared memory size per block
        stream2
    ));

    // middle pipeline stages
    for (uint step_id = 1; step_id < flash_sched -> stage_n - 1; step_id ++){
        prev_recvbuff = cur_recvbuff;
        cur_recvbuff = (step_id % 2 == 1) ? buf->internode_buffer2 : buf->internode_buffer1;
        cur_sync = (step_id % 2 == 1) ? buf->sync_signal2 : buf->sync_signal1;
        prev_sync = (step_id % 2 == 1) ? buf->sync_signal1 : buf->sync_signal2;
        prev_p2p = cur_p2p;
        cur_p2p = &(flash_sched -> dev_stages_internode[step_id]);
        cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[step_id]);

        send_chunk_param = &(flash_sched -> dev_stages_send_chunks[step_id]);
        void * internode_cur_stage[] = {
            &this_rank,
            &buf->send_buffer,
            &cur_recvbuff,
            &cur_sync,
            &cur_p2p,
            &send_chunk_param
        };
        NVSHMEMCHECK(nvshmemx_collective_launch(
            (void *) &flash_internode_p2p_chunk,
            dim3(1), 
            dim3(32),
            internode_cur_stage,
            0,
            stream1
        ));

        // wait_stage_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p);

        void * stage_redistribute_args[] = {
            &this_rank,
            &local_rank_n,
            &buf->share_mem.out_buffer.remote_data,
            &prev_recvbuff,
            &cur_redistribute,
            &k_sz -> min_tx_sz_at_a_time
        };

        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&flash_intranode_alltoall_redistribute,
                k_sz -> grid_dim,
                k_sz -> block_dim, 
                stage_redistribute_args,
                k_sz ->min_tx_sz_at_a_time,      // shared memory size per block
                stream2
            ));   
    }

    // final pipeline stage - final data redistribution
    prev_recvbuff = cur_recvbuff;
    prev_sync = cur_sync;
    prev_p2p = cur_p2p;
    cur_redistribute = &(flash_sched -> dev_stages_intra_redistribute[flash_sched -> stage_n - 1]);
    recv_chunk_param = &(flash_sched -> dev_stages_recv_chunks[flash_sched -> stage_n - 1]);

    const uint32_t chunk_n = flash_sched->host_stages_recv_chunks[flash_sched->stage_n - 1].chunk_n;
    for (uint chunkid = 0; chunkid < chunk_n; chunkid++){
        wait_chunk_ready<<<1, 1, 0, stream2>>>(prev_sync, prev_p2p, recv_chunk_param, chunkid);
        
        void * final_redistribute_chunk_args[] = {
            &this_rank,
            &local_rank_n,
            &buf->share_mem.out_buffer.remote_data,
            &prev_recvbuff,
            &cur_redistribute,
            &recv_chunk_param,
            &chunkid,
            &k_sz -> min_tx_sz_at_a_time
        };
        CUDACHECK(cudaLaunchCooperativeKernel(
            (void *)&flash_intranode_alltoall_redistribute_chunk,
                k_sz -> grid_dim,
                k_sz -> block_dim, 
                final_redistribute_chunk_args,
                k_sz ->min_tx_sz_at_a_time,      // shared memory size per block
                stream2
        ));
    }

    CUDACHECK(cudaStreamSynchronize(stream1));
    CUDACHECK(cudaStreamSynchronize(stream2));
    return NVSHMEMX_SUCCESS;
}

#if VERIFY_BUFFER == 1
size_t first_diff(const void* a, const void* b, size_t n) {
    const unsigned int* p = (const unsigned int*)a;
    const unsigned int* q = (const unsigned int*)b;
    for (size_t i = 0; i < n / sizeof(int); i++) {
        if (p[i] != q[i]) return i;  // return index of first difference
    }
    return n;  // return n if identical
}

void verify_flash_buffers(struct flash_buffer_ptr_t * buf, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n){
    void * host_recvbuff = malloc(param -> out_total_sz);
    memset(host_recvbuff, 0,  param -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, buf -> buffer_out, param -> out_total_sz, cudaMemcpyDeviceToHost));
    printf("Rank %u verifying out buffer - correct: %u\n", this_rank, (0 == memcmp(host_recvbuff, buf->buffer_verify, param->out_total_sz)));
    free(host_recvbuff);
}


void verify_flash_intra_alltoall_buffers(struct flash_buffer_ptr_t * bufs, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n){
    void * host_recvbuff = malloc(param -> out_total_sz);
    memset(host_recvbuff, 0,  param -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, bufs -> buffer_out, param -> out_total_sz, cudaMemcpyDeviceToHost));
    for (uint i = 0; i < rank_n_per_node; i++){
        uint src_rank = (this_rank / rank_n_per_node) * rank_n_per_node + i;
        printf("FLASH Intra verify: [RANK %u -> Rank %u] verifying from %lu to %lu (sz: %lu B)- correctness: %u\n", src_rank, this_rank, param->out_disp_per_rank[src_rank], param->out_disp_per_rank[src_rank] + param->out_sz_per_rank[src_rank], param->out_sz_per_rank[src_rank], (0 == memcmp((char*)host_recvbuff + param->out_disp_per_rank[src_rank], (char*)bufs->buffer_verify + param->out_disp_per_rank[src_rank], param->out_sz_per_rank[src_rank])));
    }
    free(host_recvbuff);
}
void verify_flash_inter_alltoall_buffers(struct flash_buffer_ptr_t * bufs, struct flash_buffer_sz_params_t * param, uint this_rank, uint rank_n_per_node, uint rank_n){
    void * host_recvbuff = malloc(param -> out_total_sz);
    memset(host_recvbuff, 0,  param -> out_total_sz);
    CUDACHECK(cudaMemcpy(host_recvbuff, bufs -> buffer_out, param -> out_total_sz, cudaMemcpyDeviceToHost));
    for (uint i = 0; i < rank_n; i++){
        if (i / rank_n_per_node == this_rank / rank_n_per_node) continue;
        int * vbuf = (int*)((char*)bufs->buffer_verify + param->out_disp_per_rank[i]);
        int * rbuf = (int*)((char*)host_recvbuff + param->out_disp_per_rank[i]);
        uint64_t diff_idx = first_diff((char*)host_recvbuff + param->out_disp_per_rank[i], (char*)bufs->buffer_verify + param->out_disp_per_rank[i], param->out_sz_per_rank[i]);
        if (diff_idx < param->out_sz_per_rank[i]){
            printf("FLASH Inter verify: [RANK %u -> Rank %u] verifying from %lu to %lu (sz: %lu B), correctness: %u, first-diff: %lu, first-diff int: %08x, verify: %08x\n", i, this_rank, param->out_disp_per_rank[i], param->out_disp_per_rank[i] + param->out_sz_per_rank[i], param->out_sz_per_rank[i], (0 == memcmp((char*)host_recvbuff + param->out_disp_per_rank[i], (char*)bufs->buffer_verify + param->out_disp_per_rank[i], param->out_sz_per_rank[i])), diff_idx * sizeof(int), rbuf[diff_idx], vbuf[diff_idx]);
        }
    }
    // if (this_rank == 0){
    //     printf("BUFFER OUT COTENT:\n");
    //     for (uint i = 0; i < rank_n; i ++){
    //         int * rbuf = (int*)((char*)host_recvbuff + param->out_disp_per_rank[i]);
    //         printf("From RANK %u: ", i);
    //         for (uint k = 0; k < param->out_sz_per_rank[i] / sizeof(int); k++){
    //             printf("%08x|", rbuf[k]);
    //         }
    //         printf("\n");
    //     }
    // }

    free(host_recvbuff);

    // verify balance send
    // uint local_rank_id = this_rank % rank_n_per_node;
    // if (param -> balance_send_param.total_sz > 0){
    //     uint8_t * host_lbsend = (uint8_t *) malloc(param -> balance_send_param.total_sz);
    //     uint8_t * host_lbsend_actual = (uint8_t *) malloc(param -> balance_send_param.total_sz);
    //     memset(host_lbsend, 0, param -> balance_send_param.total_sz);
    //     memset(host_lbsend_actual, 0, param -> balance_send_param.total_sz);
    //     for (uint local_gpu = 0; local_gpu < rank_n_per_node; local_gpu ++){
    //         for (uint dst_gpu = 0; dst_gpu < rank_n_per_node; dst_gpu ++){
    //             for (uint s = 0; s < rank_n / rank_n_per_node; s++){
    //                 uint dst_global_rank = s * rank_n_per_node + dst_gpu;
    //                 uint64_t disp = param -> balance_send_param.layout[local_gpu].disp[dst_global_rank];
    //                 uint64_t sz = param -> balance_send_param.layout[local_gpu].sz[dst_global_rank];
    //                 uint64_t offset = param -> balance_send_param.layout[local_gpu].data_offset[dst_global_rank];
    //                 // assert(sz % sizeof(int32_t) == 0);
    //                 // assert(disp % sizeof(int32_t) == 0);
    //                 // assert(offset % sizeof(int32_t) == 0);               
    //                 for (uint64_t z = 0; z < sz / sizeof(int32_t); z ++){
    //                     int32_t unique_data =  ((this_rank & 0xff) << 24) + ((dst_global_rank & 0xff) << 16) + ((z + offset / sizeof(int32_t)) & 0xffff);
    //                     int32_t * bb = (int32_t *) host_lbsend;
    //                     bb[disp / sizeof(int32_t) + z] = unique_data;
    //                 }
    //             }
    //         }
    //     }
    //     CUDACHECK(cudaMemcpy(host_lbsend_actual, bufs->balance_send_buffer, param -> balance_send_param.total_sz, cudaMemcpyDeviceToHost));
    //     printf("FLASH BALANCE SEND verify: Rank %u, correctness: %u\n", this_rank, (0 == memcmp(host_lbsend_actual, host_lbsend, param -> balance_send_param.total_sz)));
    //     free(host_lbsend);
    //     free(host_lbsend_actual);
    // }

    // // verify send buffer
    // uint8_t * host_sendbuff = (uint8_t *) malloc(param -> send_param.total_sz);
    // uint8_t * host_sendbuff_actual = (uint8_t *) malloc(param -> send_param.total_sz);
    // memset(host_sendbuff, 0, param -> send_param.total_sz);
    // memset(host_sendbuff_actual, 0, param -> send_param.total_sz);
    // for (uint i = 0; i < rank_n; i++){
    //     uint64_t disp = param -> send_param.layout[i].src_gpu_disp[local_rank_id];
    //     uint64_t sz = param -> send_param.layout[i].src_gpu_sz[local_rank_id];
    //     uint64_t offset = param -> send_param.layout[i].data_offset[local_rank_id];
    //     assert(sz % sizeof(int32_t) == 0);
    //     assert(disp % sizeof(int32_t) == 0);
    //     assert(offset % sizeof(int32_t) == 0);

    //     for (uint64_t j = 0; j < sz / sizeof(int32_t); j++){
    //         int32_t unique_data = ((this_rank & 0xff) << 24) + ((i & 0xff) << 16) + ((j + offset / sizeof(int32_t)) & 0xffff);
    //         int32_t * sb = (int32_t *) host_sendbuff;
    //         sb[disp / sizeof(int32_t) + j] = unique_data;
    //     }
    // }
    // CUDACHECK(cudaMemcpy(host_sendbuff_actual, bufs->send_buffer, param -> send_param.total_sz, cudaMemcpyDeviceToHost));
    // printf("FLASH SENDBUFF verify: Rank %u, correctness: %u\n", this_rank, (0 == memcmp(host_sendbuff_actual, host_sendbuff, param -> send_param.total_sz)));

    // free(host_sendbuff);
    // free(host_sendbuff_actual);

    // // verify lb recv
    // if(param -> balance_recv_param.total_sz > 0){
    //     uint8_t * host_lbrecv = (uint8_t *) malloc(param -> balance_recv_param.total_sz);
    //     memset(host_lbrecv, 0, param -> balance_recv_param.total_sz);
    //     CUDACHECK(cudaMemcpy(host_lbrecv, bufs->balance_recv_buffer, param -> balance_recv_param.total_sz, cudaMemcpyDeviceToHost));
    //     int * rbuf = (int*) host_lbrecv;
    //     printf("FLASH LBRECV verify: Rank %u, first int: %08x\n", this_rank, rbuf[0]);
    //     free(host_lbrecv);
    // }
}
#endif


void set_kernel_sharemem_sz( struct kernel_sz_t * ksz){
    uint64_t kernel_share_memsize = ksz -> min_tx_sz_at_a_time + 16;
    CUDACHECK(cudaFuncSetAttribute(intranode_alltoallv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize ));
    CUDACHECK(cudaFuncSetAttribute(spreadout_alltoallv_intranode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize ));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_no_signal, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_simple, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_redistribute, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_redistribute_no_signal, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_redistribute_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
    CUDACHECK(cudaFuncSetAttribute(flash_intranode_alltoall_redistribute_simple, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel_share_memsize));
}