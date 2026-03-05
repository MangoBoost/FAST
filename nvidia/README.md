# FAST All-to-All (NVIDIA + NVSHMEM)

This directory contains the NVIDIA implementation of FAST all-to-all.

## Folder Structure

```text
fastalltoall/
├── CMakeLists.txt
├── README.md
├── alltoall_nvshmem.cpp
├── flash_tester.py
├── flash_utils.py
└── src/
    ├── fast_alltoall/
    │   ├── alltoall_algorithm.cc
    │   ├── alltoall_global_scheduler.cc
    │   ├── alltoall_local_scheduler.cc
    │   ├── alltoall_matrix.cc
    │   └── flash_alltoall_nvshmem.cu
    └── include/
        ├── registration.h
        └── fast_alltoall/
            ├── alltoall_algorithm.h
            ├── alltoall_define.h
            ├── alltoall_global_scheduler.h
            ├── alltoall_local_scheduler.h
            ├── alltoall_matrix.h
            ├── atomic.cuh
            └── flash_alltoall_nvshmem.h
```

## What Lives Where

- Core runtime implementation is in `src/fast_alltoall/flash_alltoall_nvshmem.cu`.
- `alltoall_nvshmem.cpp` is the host/test entrypoint:
  - process-group/NVSHMEM setup
  - workload generation/broadcast
  - schedule invocation
  - buffer setup/open/close
  - perf testing and optional buffer verification
- Python binding/test path:
  - `flash_utils.py` loads `libflash.so`
  - `flash_tester.py` drives distributed tests

## Scheduling Layers

1. Global volume-level schedule (`scheduling_result_t`)
- Defined in `src/include/fast_alltoall/alltoall_global_scheduler.h`.
- Stores high-level schedule objects for the full system:
  - load-balance metadata
  - per-step inter-server mapping/order
  - cross-node volumes and redistribution volumes
- This level is close to the simulation logic and represents transfer structure/volume.

2. Per-GPU communication schedule (`scheduling_result_gpu_t`)
- Also in `alltoall_global_scheduler.h`.
- Converts global schedule into one GPU's send/recv/memcpy actions.
- Still host-side metadata and not yet kernel-block/chunk ready.

3. Kernel-ready FAST schedule (`flash_schedule_this_gpu_t`)
- Also in `alltoall_global_scheduler.h`.
- Contains per-stage device metadata used directly by kernels:
  - internode p2p stage params
  - intra-node redistribute params
  - chunk thresholds/disp/sizes
  - intrinsic and balance intra-node plans
- This is the schedule consumed at runtime by FAST kernels.

4. Runtime buffers (`flash_buffer_ptr_t`)
- Defined in `src/include/fast_alltoall/flash_alltoall_nvshmem.h`.
- Contains FAST runtime buffers, including temporary/pipeline buffers:
  - `send_buffer`
  - `internode_buffer1/2` (ping-pong)
  - `balance_send_buffer`, `balance_recv_buffer`
  - completion/sync/credit buffers
  - final `buffer_out`


## End-to-End Runtime Flow

1. Demand matrix is created/broadcast in `alltoall_nvshmem.cpp`.
2. `flash_scheduler(...)` builds schedule metadata:
- decomposition + global pipeline plan
- per-GPU schedule
- kernel-ready FLASH schedule + buffer-size plan
3. `init_flash_buffer(...)` allocates runtime/NVSHMEM buffers.
4. `launch_flash_alltoallv(...)` executes staged transfer.

## Build

From `nvidia/`:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parents[1] / "share" / "cmake")')" \
  -DNVSHMEM_ROOT="/path/to/nvshmem"

cmake --build build -j
```

Output shared library:

```text
nvidia/libflash.so
```


## Multi-Node Launch Script

Use this `sbatch` template for multi-node launch and adjust partition/resources for your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=fastalltoall
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=./log/flash_%j.out
#SBATCH --error=./log/flash_%j.err
#SBATCH --exclusive
#SBATCH --time=24:00:00
# Optional, cluster-specific:
#SBATCH --qos=normal
#SBATCH --exclude=nodeA,nodeB

set -euo pipefail

export NVSHMEM_IB_ENABLE_IBGDA=true
# Optional NVSHMEM tuning:
# export NVSHMEM_DEBUG=WARN
# export NVSHMEM_SYMMETRIC_SIZE=2G
# export NVSHMEM_IBGDA_NUM_DCI=32
# export NVSHMEM_IBGDA_NUM_DCT=32
# export NVSHMEM_IBGDA_NUM_RC_PER_PE=32

GPUS_PER_NODE=8
NUM_NODES=${SLURM_NNODES}

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)
RDZV_PORT=29500

mkdir -p log
cd "${SLURM_SUBMIT_DIR}"

DISTRIBUTED_ARGS=(
  --nproc_per_node="${GPUS_PER_NODE}"
  --nnodes="${NUM_NODES}"
  --rdzv_id="${SLURM_JOB_ID}"
  --rdzv_backend=c10d
  --rdzv_endpoint="${head_node_ip}:${RDZV_PORT}"
)

srun torchrun "${DISTRIBUTED_ARGS[@]}" flash_tester.py
```


##  Notes

- `launch_flash_alltoallv` is the optimized FAST pipeline path.
- `launch_alltoallv` is a simpler spreadout/fanout baseline path.
- The decomposition/schedule-generation logic is shared in spirit with `simulation/`; understanding that first is the easiest entry point before diving into buffer layout and kernel staging.
