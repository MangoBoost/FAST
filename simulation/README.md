# FAST Simulator

This folder contains the simulator for **FAST**, an efficient scheduler for all-to-all communication.

## Folder structure

```text
simulation/
├── algorithm/
│   ├── all2all.h/.cpp      # FastAll2All decomposition (server-level permutation sets)
│   └── matrix.h/.cpp       # Matrix utilities
├── scheduler/
│   ├── local.h/.cpp        # Intra-server scheduling
│   ├── global.h/.cpp       # End-to-end FAST pipeline and runtime model
│   └── config.h/.cpp       # Link model + baselines
├── plot/                   # Plot scripts and generated figures
├── main.cpp                # Entry point: choose which benchmark/test to run
├── test.h/.cpp             # Workload generation + benchmark drivers
├── define.h                # Global constants (for example BENCHMARK_DIR)
└── Makefile                # Build and run targets
```

## Build and run

From the `simulation/` directory:

```bash
make clean
make build
make run
```

Binary path:

```text
simulation/build/all2all_simulator
```

## Workload and testbed setup (`test.cpp`)

- **Workload (traffic matrix)** is generated in `test.cpp` (random matrix per trial, diagonal forced to 0).
- **Testbed scale** is controlled by `server_n` and `gpu_n` in `FastAll2AllTester`.

Main configuration entry is in `main.cpp`:

```cpp
FastAll2AllTester simulator(2, 16, 20, false, ETHER400, H100);
simulator.server_gpu_number_benchmark(H100, ETHER100);
```

Constructor arguments are:

```text
FastAll2AllTester(server_n, gpu_n, test_times, enable_scaling, inter_link_type, intra_link_type)
```

`test.cpp` provides benchmark methods such as:
- `run()`
- `server_gpu_number_benchmark(...)`
- `fabric_speed_benchmark()`
- `topology_benchmark(...)`
- `skewness_benchmark(...)`
- `transfer_size_benchmark(...)`

## Where FAST scheduling results are stored

Core result container is in `algorithm/all2all.h`:

- `class FastAll2All`
- `vector<PermutationSet> p_sets;`

After calling:

1. `to_scaled_doubly_stochastic_matrix()` (and optional `to_scaled_matrix(...)`)
2. `decompose()`

`p_sets` stores the Birkhoff decomposition result:

- each `PermutationSet` holds `mp` (row server -> destination server mapping)
- `get_freq()` gives the transfer frequency/weight for that permutation

`GlobalScheduler::pipeline2(...)` consumes `all2all.p_sets` to build inter-server transfer order and intra-server dispatch schedule.

## Benchmark outputs

Benchmark functions in `test.cpp` write text outputs under `benchmark/` (for example speedup/time data).
Create this directory before running benchmarks:

```bash
mkdir -p benchmark
```
