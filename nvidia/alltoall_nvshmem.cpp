// #include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fast_alltoall/alltoall_global_scheduler.h>
#include <fast_alltoall/alltoall_local_scheduler.h>
#include <fast_alltoall/flash_alltoall_nvshmem.h>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmemx.h>
#include <nvshmem.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdlib>
#include <torch/library.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <vector>
#include <registration.h>


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
            workload[i * nrank + j] = mem_align(zipf_inverse_cdf_fast(s, p, bound)) * sizeof(int32_t);
        }
    }
    // clean the diagnal
    // for (uint i = 0; i < nrank; i++){
    //      workload[i * nrank + i] = 0;
    // }
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

void fixed_distribution(uint64_t * workload, uint nrank, uint64_t bound){
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = mem_align(bound) * sizeof(int32_t);
            workload[i * nrank + j] = bound * sizeof(int32_t);
        }
    }
    // clean the diagnal
    // for (uint i = 0; i < nrank; i++){
    //      workload[i * nrank + i] = 0;
    // }
}

void uniform_distribution(uint64_t * workload, uint nrank, uint64_t bound){
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = (rand() % bound) * sizeof(int32_t);
            workload[i * nrank + j] = mem_align(rand() % bound);
            // workload[i * nrank + j] = mem_align(bound) * sizeof(int32_t);
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
            // if (i == j) {
            //     workload[i * nrank + j] = 0;
            //     continue;
            // }
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = rand() % 1024;
            total_sz += workload[i * nrank + j];
        }
    }
    uint64_t multiplier = per_gpu_sz * (nrank) * nrank / total_sz;
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            // workload[i * nrank + j] = bound;
            workload[i * nrank + j] = mem_align(workload[i * nrank + j] * multiplier);
        }
    }
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

struct perf_test_ret_t perf_flash(uint warmup_iters, 
    uint perf_iters, 
    struct flash_buffer_ptr_t * data_buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param, 
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1, 
    cudaStream_t stream2,
    cudaStream_t stream3,
    uint64_t buff_size,
    uint data_size){

    cudaEvent_t start_event, end_event;
    uint event_n = flash_sched -> stage_n + 1;
    cudaEvent_t * sync_events = (cudaEvent_t *) malloc (sizeof(cudaEvent_t) * event_n);
    for (uint i = 0; i < event_n; i++){
        CUCHECK(cuEventCreate(&sync_events[i], cudaEventDisableTiming));
    }

    CUCHECK(cuEventCreate(&start_event, CU_EVENT_DEFAULT));
    CUCHECK(cuEventCreate(&end_event, CU_EVENT_DEFAULT));

    nvshmem_barrier_all();
    for (int i = 0; i < warmup_iters; ++i) {
        NVSHMEMCHECK(launch_flash_alltoallv(
            data_buf, 
            flash_sched,
            flash_param,
            k_sz, 
            stream1,
            stream2,
            stream3,
            sync_events,
            event_n,
            data_size,
            i
        ));
    }
    // nvshmem_barrier_all();
    CUCHECK(cuEventRecord(start_event, stream1));
    for (int i = 0; i < perf_iters; ++i) {
        NVSHMEMCHECK(launch_flash_alltoallv(
            data_buf, 
            flash_sched,
            flash_param,
            k_sz, 
            stream1,
            stream2,
            stream3,
            sync_events,
            event_n,
            data_size,
            i
        ));
    }
    CUCHECK(cuEventRecord(end_event, stream1));
    CUDACHECK(cudaStreamSynchronize(stream3));
    CUDACHECK(cudaStreamSynchronize(stream2));
    CUDACHECK(cudaStreamSynchronize(stream1));
    nvshmem_quiet();
    nvshmem_barrier_all();

    float elapsed_time;
    CUCHECK(cuEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / flash_sched->info.rank_n;

    CUCHECK(cuEventDestroy(start_event));
    CUCHECK(cuEventDestroy(end_event));
    for (uint i = 0; i < event_n; i++){
        CUCHECK(cuEventDestroy(sync_events[i]));
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = 0.0, .crossnode_time = avg_time, .restore_time = 0.0};
    return r;
}

struct perf_test_ret_t perf_flash_chunk(uint warmup_iters, 
    uint perf_iters, 
    struct flash_buffer_ptr_t * data_buf,
    struct flash_schedule_this_gpu_t * flash_sched,
    struct flash_buffer_sz_params_t * flash_param, 
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1, 
    cudaStream_t stream2,
    cudaStream_t stream3,
    uint64_t buff_size){

    cudaEvent_t start_event, end_event;
    CUCHECK(cuEventCreate(&start_event, CU_EVENT_DEFAULT));
    CUCHECK(cuEventCreate(&end_event, CU_EVENT_DEFAULT));
    uint event_n = flash_sched -> stage_n + 1;
    cudaEvent_t * sync_events = (cudaEvent_t *) malloc (sizeof(cudaEvent_t) * event_n);
    for (uint i = 0; i < event_n; i++){
        CUCHECK(cuEventCreate(&sync_events[i], cudaEventDisableTiming));
    }

    nvshmem_barrier_all();
    for (int i = 0; i < warmup_iters; ++i) {
        NVSHMEMCHECK(launch_flash_alltoallv_chunk(
            data_buf, 
            flash_sched,
            flash_param,
            k_sz, 
            stream1,
            stream2,
            stream3,
            sync_events,
            event_n,
            i));
    }
    nvshmem_barrier_all();
    CUCHECK(cuEventRecord(start_event, stream1));
    for (int i = 0; i < perf_iters; ++i) {
        NVSHMEMCHECK(launch_flash_alltoallv_chunk(
            data_buf, 
            flash_sched,
            flash_param,
            k_sz, 
            stream1,
            stream2,
            stream3,
            sync_events,
            event_n,
            i));
    }
    CUCHECK(cuEventRecord(end_event, stream1));
    nvshmem_quiet();
    nvshmem_barrier_all();

    float elapsed_time;
    CUCHECK(cuEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / flash_sched->info.rank_n;

    CUCHECK(cuEventDestroy(start_event));
    CUCHECK(cuEventDestroy(end_event));
    for (uint i = 0; i < event_n; i++){
        CUCHECK(cuEventDestroy(sync_events[i]));
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = 0.0, .crossnode_time = avg_time, .restore_time = 0.0};
    return r;
}


struct perf_test_ret_t perf_fanout(uint warmup_iters, 
    uint perf_iters, 
    struct intra_then_inter_data_buffer_t * data_buf,
    struct fanout_nvshmem_buffer_t * buf,  
    struct intranode_transfer_params_t * intra_params,
    struct internode_transfer_params_t * inter_params,
    struct kernel_sz_t * k_sz,
    cudaStream_t stream1, 
    cudaStream_t stream2, 
    uint64_t buff_size, 
    uint rank_id,
    uint rank_n_per_node, 
    uint rank_n){

    cudaEvent_t start_event, end_event;

    CUCHECK(cuEventCreate(&start_event, CU_EVENT_DEFAULT));
    CUCHECK(cuEventCreate(&end_event, CU_EVENT_DEFAULT));

    nvshmem_barrier_all();
    for (int i = 0; i < warmup_iters; ++i) {
        CUDACHECK(cudaMemcpyAsync((uint8_t *) buf->send_buffer + data_buf -> intra_send_sz, (uint8_t *) data_buf->buffer_in + data_buf -> intra_send_sz, data_buf -> in_total_sz - data_buf -> intra_send_sz, cudaMemcpyDeviceToDevice, stream1));
        NVSHMEMCHECK(launch_alltoallv(
            data_buf, 
            rank_id, 
            rank_n_per_node, 
            rank_n, 
            buf, 
            intra_params, 
            inter_params,
            k_sz, 
            stream1,
            stream2));
        CUDACHECK(cudaMemcpyAsync((uint8_t *) data_buf->buffer_out + data_buf -> intra_recv_sz, (uint8_t *) buf->recv_buffer + data_buf -> intra_recv_sz, data_buf -> out_total_sz - data_buf -> intra_recv_sz, cudaMemcpyDeviceToDevice, stream1));
    }
    CUDACHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUCHECK(cuEventRecord(start_event, stream1));
    for (int i = 0; i < perf_iters; ++i) {
        NVSHMEMCHECK(launch_alltoallv(
            data_buf, 
            rank_id, 
            rank_n_per_node, 
            rank_n, 
            buf, 
            intra_params, 
            inter_params,
            k_sz, 
            stream1,
            stream2));
    }
    CUCHECK(cuEventRecord(end_event, stream1));
    CUDACHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();


    float elapsed_time;
    CUCHECK(cuEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size * (1e-6) / avg_time / rank_n;

    CUCHECK(cuEventDestroy(start_event));
    CUCHECK(cuEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time, .loadbalance_time = 0.0, .crossnode_time = avg_time, .restore_time = 0.0};
    return r;
}


at::Tensor get_nvshmem_init_id(){
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&id);
    return at::from_blob(&id, sizeof(id), at::kByte).clone();
}

struct flash_comm_t : torch::CustomClassHolder{
    int rank_id;
    int local_rank_id;
    int local_rank_n;
    int rank_n;
    int server_id;
    int server_n;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    nvshmem_team_t global_team;
    nvshmem_team_t local_team;
    struct GlobalScheduler * scheduler;

    flash_comm_t(int64_t rank, int64_t dev_n, int64_t world_size, at::Tensor uid){
        rank_id = rank;
        local_rank_n = dev_n;
        rank_n = world_size;
        server_id = rank_id / local_rank_n;
        local_rank_id = rank_id % local_rank_n;
        server_n = rank_n / local_rank_n;
        CUDACHECK(cudaSetDevice(rank_id % local_rank_n));
        int least, greatest;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        CUDACHECK(cudaStreamCreateWithPriority(&stream3, cudaStreamNonBlocking, greatest)); // highest prio
        CUDACHECK(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, greatest));
        CUDACHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, least));
        // CUDACHECK(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));
        TORCH_CHECK(uid.device().is_cpu(), "uid must be a CPU tensor");
        TORCH_CHECK(uid.scalar_type() == at::kByte, "uid must be a byte tensor");
        TORCH_CHECK(
            uid.numel() == sizeof(nvshmemx_uniqueid_t),
            "Invalid unique id size (expected ",
            sizeof(nvshmemx_uniqueid_t),
            ", got ",
            uid.numel(),
            ")"
        );
        nvshmemx_uniqueid_t id;
        std::memcpy(&id, uid.data_ptr(), sizeof(id));
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_set_attr_uniqueid_args(rank_id, rank_n, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        global_team = NVSHMEM_TEAM_WORLD;
        int nvshmem_global_rank = nvshmem_team_my_pe(global_team);
        int nvshmem_global_nranks = nvshmem_team_n_pes(global_team);
        // create local (ranks within the same node) communication group
        nvshmem_team_split_strided(global_team, server_id * local_rank_n, 1, local_rank_n, nullptr, 0, &local_team);
        int nvshmem_local_rank = nvshmem_team_my_pe(local_team);
        int nvshmem_local_nranks = nvshmem_team_n_pes(local_team);
        //enable peer-to-peer access
        for (uint i = 0; i < local_rank_n; i++){
            if (i == rank_id % local_rank_n) continue;        
            CUDACHECK(cudaDeviceEnablePeerAccess(i, 0));
        }
        printf("FLASH is initialized: Torch Rank %d/%d, dev_n: %d, NVSHMEM Gobal Rank %d/%d, NVSHMEM Local Rank: %d/%d \n", rank_id, rank_n, local_rank_n, nvshmem_global_rank, nvshmem_global_nranks, nvshmem_local_rank, nvshmem_local_nranks);
        CUDACHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        printf("Rank %d Constructor NVSHMEM status: %d\n", rank_id, nvshmemx_init_status());
        scheduler = NULL;
    }

    void alltoall_cpu(nvshmem_team_t team, void * host_src_buffer, void * host_dst_buffer, int rank_n, size_t size_per_rank){
        // for exchaning meta data
        void * nvshmem_src_buffer = nvshmem_malloc(size_per_rank * rank_n);
        void * nvshmem_dst_buffer = nvshmem_malloc(size_per_rank * rank_n);
        CUDACHECK(cudaMemcpy(nvshmem_src_buffer, host_src_buffer, size_per_rank * rank_n, cudaMemcpyHostToDevice));
        nvshmem_alltoallmem(team, nvshmem_dst_buffer, nvshmem_src_buffer, size_per_rank);
        nvshmem_quiet();
        CUDACHECK(cudaMemcpy(host_dst_buffer, nvshmem_dst_buffer, rank_n * size_per_rank, cudaMemcpyDeviceToHost));
        nvshmem_free(nvshmem_src_buffer);
        nvshmem_free(nvshmem_dst_buffer);
    }

    void broadcast_cpu(nvshmem_team_t team, void * host_src_buffer, void * host_dst_buffer, size_t size, int root){
        void * nvshmem_src_buffer = nvshmem_malloc(size);
        void * nvshmem_dst_buffer = nvshmem_malloc(size);        
        CUDACHECK(cudaMemcpy(nvshmem_src_buffer, host_src_buffer, size, cudaMemcpyHostToDevice));
        NVSHMEMCHECK(nvshmem_broadcastmem(team, nvshmem_dst_buffer, nvshmem_src_buffer, size, root));
        nvshmem_quiet();
        CUDACHECK(cudaMemcpy(host_dst_buffer, nvshmem_dst_buffer, size, cudaMemcpyDeviceToHost));
        nvshmem_free(nvshmem_dst_buffer);
        nvshmem_free(nvshmem_src_buffer);
    }


    void open_remote_memory(struct shared_memory_t * smem, uint local_rank_id, uint local_rank_n){
        //collect memory handles from all the ranks within the same server
        nvshmem_barrier(global_team);
        uint8_t * host_src_buffer = (uint8_t *) malloc(local_rank_n * sizeof(cudaIpcMemHandle_t));
        uint8_t * host_dst_buffer = (uint8_t *) malloc(local_rank_n * sizeof(cudaIpcMemHandle_t));
        memset(host_dst_buffer, 0, local_rank_n * sizeof(cudaIpcMemHandle_t));
        for (uint i = 0; i < local_rank_n; i++){
            // local handle will be sent to every other rank within the same node
            memcpy(host_src_buffer + i * sizeof(cudaIpcMemHandle_t), &smem->local_handle, sizeof(cudaIpcMemHandle_t));
        }
        alltoall_cpu(local_team, host_src_buffer, host_dst_buffer, local_rank_n, sizeof(cudaIpcMemHandle_t));
        cudaIpcMemHandle_t * remote_handles = (cudaIpcMemHandle_t *)host_dst_buffer;
        for (uint i = 0; i < local_rank_n; i ++){
            if (i == local_rank_id) {
                smem->host_remote_data[i] = smem->local_data;
                continue;
            }
            smem->remote_handle[i] = remote_handles[i];
            CUDACHECK(cudaIpcOpenMemHandle(&smem->host_remote_data[i], smem->remote_handle[i], cudaIpcMemLazyEnablePeerAccess));
        }
        CUDACHECK(cudaMemcpy(smem->remote_data, smem->host_remote_data, sizeof(void *) * local_rank_n, cudaMemcpyHostToDevice));
        free(host_src_buffer);
        free(host_dst_buffer);
    }

    void close_remote_memory(struct shared_memory_t * smem, uint local_rank_id, uint local_rank_n){
        for (uint i = 0; i < local_rank_n; i ++){
            if (i == local_rank_id) continue;
            CUDACHECK(cudaIpcCloseMemHandle(smem->host_remote_data[i]));
        }
    }

    void test_basic_alltoall(){
        uint64_t * workload_in_bytes = new uint64_t[rank_n * rank_n];
        if (rank_id == 0) uniform_distribution(workload_in_bytes, rank_n, pow(2,23));
        broadcast_cpu(global_team, workload_in_bytes, workload_in_bytes, rank_n * rank_n * sizeof(uint64_t), 0);

        struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload_in_bytes, rank_n);
        uint64_t buff_size = workload_max_sum.sum;    // bytes
        if (rank_id == 0) print_matrix(workload_in_bytes, rank_n, rank_n);

        struct kernel_sz_t kernel_sz = kernel_sz_for_workload(32, 32, 8);  // warp_n, block_n, TX_BURST
        struct intra_then_inter_data_buffer_t data_buffer = init_data_buffer(workload_in_bytes, rank_id, local_rank_n, rank_n, kernel_sz.block_n);
        // struct fanout_buffer_params_t params = init_fanout_params(workload_in_bytes, rank_id, local_rank_n, rank_n, &kernel_sz);
        struct internode_transfer_params_t inter_params = init_internode_transfer_params(&data_buffer, rank_id, local_rank_n, rank_n);
        struct intranode_transfer_params_t intra_params = init_intranode_transfer_params(&data_buffer, rank_id, local_rank_n, rank_n, &kernel_sz);
        // struct fanout_data_buffer_t data_buffer = init_fanout_data_buffer(workload_in_bytes, rank_id, local_rank_n, rank_n, kernel_sz.block_n);
        open_remote_memory(&data_buffer.share_mem.recv_buffer, local_rank_id, local_rank_n);
        std::cout << "IPC enabled at local RANK " << local_rank_id << std::endl;
        struct fanout_nvshmem_buffer_t nvshmem_buffer = init_fanout_nvshmem_buffer(workload_in_bytes, rank_id, local_rank_n, rank_n, kernel_sz.block_n);
        printf("[RANK %u]: grid dim: (%u, %u, %u), block dim: (%u, %u, %u), min_tx_at_a_time: %lu, share mem size: %lu\n", rank_id, kernel_sz.grid_dim.x, kernel_sz.grid_dim.y, kernel_sz.grid_dim.z, kernel_sz.block_dim.x, kernel_sz.block_dim.y, kernel_sz.block_dim.z, kernel_sz.min_tx_sz_at_a_time, kernel_sz.min_tx_sz_at_a_time);
        set_kernel_sharemem_sz(&kernel_sz);
        struct perf_test_ret_t fanout = perf_fanout(20, 20, &data_buffer, &nvshmem_buffer, &intra_params, &inter_params, &kernel_sz, stream1, stream2, buff_size, rank_id, local_rank_n, rank_n);

        if (rank_id == 0) std::cout << "fanout algbw: " << fanout.algbw << " GBps" <<std::endl
                            << "total time: " << fanout.time << " ms" << std::endl;
#if VERIFY_BUFFER == 1
        verify_intra_alltoall_buffers(&data_buffer, rank_id, local_rank_n, rank_n);
        verify_inter_alltoall_buffers(&data_buffer, rank_id, local_rank_n, rank_n);
#endif

        // free_fanout_params(&params);
        free_intranode_transfer_params(&intra_params);
        free_internode_transfer_params(&inter_params);
        // close_remote_memory(&nvshmem_buffer.shared_recv_buffer, local_rank_id, local_rank_n);
        // free_fanout_nvshmem(&nvshmem_buffer);
        close_remote_memory(&data_buffer.share_mem.recv_buffer, local_rank_id, local_rank_n);
        // free_fanout_data_buffer(&data_buffer);
        free_data_buffer(&data_buffer);
        delete[] workload_in_bytes;
    }

    
    void schedule(uint64_t * workload, uint block_n, uint64_t block_tx_sz, uint data_size){
        // if (rank == 0) print_matrix(workload, nrank, nrank);
        if (server_n > 1){
            if (scheduler == NULL){
                scheduler = new struct GlobalScheduler;
                init_global_scheduler(scheduler, server_n, local_rank_n, workload, rank_id, block_n, block_tx_sz, data_size);
                flash_scheduler(scheduler);
            }else{
                update_global_scheduler(scheduler, workload);
                flash_scheduler(scheduler);
            }
        }
    }

    void test_flash_alltoall(){
        uint64_t * workload_in_bytes = new uint64_t[rank_n * rank_n];
        if (rank_id == 0) uniform_distribution(workload_in_bytes, rank_n, pow(2,23));
        // if (rank_id == 0) zipf_distribution(workload_in_bytes, 0.8, rank_n, pow(2,23));
        broadcast_cpu(global_team, workload_in_bytes, workload_in_bytes, rank_n * rank_n * sizeof(uint64_t), 0);

        uint data_size = 8;
        struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload_in_bytes, rank_n);
        uint64_t buff_size = workload_max_sum.sum * data_size;    // bytes
        if (rank_id == 0) print_matrix(workload_in_bytes, rank_n, rank_n);

        struct kernel_sz_t kernel_sz = kernel_sz_for_workload(16, 32, 8);  // warp_n, block_n, TX_BURST
        schedule(workload_in_bytes, kernel_sz.block_n, kernel_sz.min_tx_sz_at_a_time, data_size);
        struct flash_buffer_ptr_t data_buf = init_flash_buffer(scheduler->flash_buffer_sz_params, rank_id, local_rank_n, rank_n, kernel_sz.block_n, data_size);
        // open memory handle at remote GPU 
        open_remote_memory(&data_buf.share_mem.out_buffer, local_rank_id, local_rank_n);
        open_remote_memory(&data_buf.share_mem.balance_recv_buffer, local_rank_id, local_rank_n);
        open_remote_memory(&data_buf.share_mem.recv_complete_signal, local_rank_id, local_rank_n);
        // std::cout << "FLASH IPC enabled at local RANK " << local_rank_id << std::endl;
        set_kernel_sharemem_sz(&kernel_sz);
        struct perf_test_ret_t flash = perf_flash(50, 50, &data_buf, scheduler -> flash_sched , scheduler -> flash_buffer_sz_params, &kernel_sz, stream1, stream2, stream3, buff_size, data_size);
        if (rank_id == 0) std::cout <<"per-GPU size:" << buff_size / rank_n / 1e6<< " MB, flash algbw: " << flash.algbw << " GBps" <<std::endl
                    << "total time: " << flash.time << " ms" << std::endl;

        // struct perf_test_ret_t flash_chunk = perf_flash_chunk(20, 20, &data_buf, scheduler -> flash_sched, scheduler -> flash_buffer_sz_params, &kernel_sz, stream1, stream2, stream3, buff_size);
        // if (rank_id == 0) std::cout << "flash chunk algbw: " << flash_chunk.algbw << " GBps" <<std::endl
        //                     << "total time: " << flash_chunk.time << " ms" << std::endl;

#if VERIFY_BUFFER == 1
        verify_flash_buffers(&data_buf, scheduler -> flash_buffer_sz_params, rank_id, local_rank_n, rank_n);
        // verify_flash_intra_alltoall_buffers(&data_buf, scheduler -> flash_buffer_sz_params, rank_id, local_rank_n, rank_n);
        // verify_flash_inter_alltoall_buffers(&data_buf, scheduler -> flash_buffer_sz_params, rank_id, local_rank_n, rank_n);
#endif
        close_remote_memory(&data_buf.share_mem.recv_complete_signal, local_rank_id, local_rank_n);
        close_remote_memory(&data_buf.share_mem.balance_recv_buffer, local_rank_id, local_rank_n);
        close_remote_memory(&data_buf.share_mem.out_buffer, local_rank_id, local_rank_n);
        free_flash_buffer(&data_buf);
        delete[] workload_in_bytes;
    }

    // void test_flash_under_different_transfer_sz(int64_t power, int64_t testid){
    //     uint64_t * workload_in_bytes = new uint64_t[rank_n * rank_n];
    //     if (rank_id == 0) uniform_distribution(workload_in_bytes, rank_n, pow(2,power));
    //     broadcast_cpu(global_team, workload_in_bytes, workload_in_bytes, rank_n * rank_n * sizeof(uint64_t), 0);
    //     struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload_in_bytes, rank_n);
    //     uint64_t buff_size = workload_max_sum.sum;    // bytes
    //     struct kernel_sz_t kernel_sz = kernel_sz_for_workload(32, 32, 8);  // warp_n, block_n, TX_BURST
    //     schedule(workload_in_bytes, kernel_sz.block_n, kernel_sz.min_tx_sz_at_a_time);
    //     struct flash_buffer_ptr_t data_buf = init_flash_buffer(scheduler->flash_buffer_sz_params, rank_id, local_rank_n, rank_n, kernel_sz.block_n);
    //     // open memory handle at remote GPU 
    //     open_remote_memory(&data_buf.share_mem.out_buffer, local_rank_id, local_rank_n);
    //     open_remote_memory(&data_buf.share_mem.balance_recv_buffer, local_rank_id, local_rank_n);
    //     open_remote_memory(&data_buf.share_mem.recv_complete_signal, local_rank_id, local_rank_n);
    //     // std::cout << "FLASH IPC enabled at local RANK " << local_rank_id << std::endl;
    //     set_kernel_sharemem_sz(&kernel_sz);
    //     struct perf_test_ret_t flash = perf_flash(20, 20, &data_buf, scheduler -> flash_sched , scheduler -> flash_buffer_sz_params, &kernel_sz, stream1, stream2, stream3, buff_size);
    //     if (rank_id == 0) std::cout << buff_size / rank_n / 1e6 << " MB, "  << testid << "-th workload, " << flash.algbw << " GBps, " << flash.time << " ms" << std::endl;
    //     // verify_flash_buffers(&data_buf, scheduler -> flash_buffer_sz_params, rank_id, local_rank_n, rank_n);
    //     close_remote_memory(&data_buf.share_mem.recv_complete_signal, local_rank_id, local_rank_n);
    //     close_remote_memory(&data_buf.share_mem.balance_recv_buffer, local_rank_id, local_rank_n);
    //     close_remote_memory(&data_buf.share_mem.out_buffer, local_rank_id, local_rank_n);
    //     free_flash_buffer(&data_buf);
    //     delete[] workload_in_bytes;
    // }


    ~flash_comm_t(){
        nvshmem_barrier_all();
        CUCHECK(cuStreamDestroy(stream1));
        CUCHECK(cuStreamDestroy(stream2));
        CUCHECK(cuStreamDestroy(stream3));
        nvshmem_team_destroy(local_team);
        nvshmem_finalize();
        printf("FLASH is freed: Torch RANK %d/%d\n", rank_id, rank_n);
        if (scheduler){
            std::cout << "cleaning scheduler" << std::endl;
            free_global_scheduler(scheduler);
            delete scheduler;
        }
    }
};


TORCH_LIBRARY(my_classes, m) {
  m.class_<flash_comm_t>("flash_comm_t")
    .def(torch::init<int64_t, int64_t, int64_t, at::Tensor>())
    .def("test_basic_alltoall", &flash_comm_t::test_basic_alltoall)
    // .def("test_flash_under_different_transfer_sz", &flash_comm_t::test_flash_under_different_transfer_sz)
    .def("test_flash_alltoall", &flash_comm_t::test_flash_alltoall);
}

void register_flash_ops(torch::Library &m) {
    m.def("get_nvshmem_init_id", &get_nvshmem_init_id);
}

TORCH_LIBRARY(flash_nvshmem, m) {
  register_flash_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
