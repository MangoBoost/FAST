This repository contains the source code and test program for the FLASH All-to-All schduler.
The repository is structured as follows:
```
|fastalltoall
|----- src                  (Source code of FLASH scheduler)
|----- result               (Some collected experiment results)
|----- flash.cpp            (FLASH Python interface)
|----- torch_tester.sh      (FLASH Python interface test program)
|----- torch_run.sh         (FLASH Python interface test program launch script)
|----- test_alltoall.cpp    (FLASH performance test program)
|----- Makefile             (Compile script)
```

# Performance Test
The performance test is purely C/C++ program.
Run the following command to compile the performance test program, which builds a `all2all_tester` in the `build` folder:
```
make clean
make compile
```
Check and update the paths of dependent libraries in `Makefile`.
Make sure all the nodes have the same version of the test program.

To launch the performance test program, run the following command:
```
	mpirun --allow-run-as-root \
    -hostfile <path-to-hostfile> \
    -map-by ppr:8:node --bind-to numa -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include <net-iface> \
    -x LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH \
    -x NCCL_SOCKET_IFNAME=<net-iface> \
    -x LD_PRELOAD=<path-to-rccl-lib>:$LD_PRELOAD \
    -x NCCL_DEBUG=WARN \
    -x NCCL_DEBUG_SUBSYS=INIT,GRAPH \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -x NCCL_IBEXT_DISABLE=1 \
    -x NCCL_PROTO=Simple  \
    bash run.sh
```
Check and update the paths of dependent libraries and files.
Make sure the MPI environment is correctly configured, you can check by using the following command:
```
mpirun --allow-run-as-root -hostfile <path-to-hostfile> hostname
```


The performance test program includes many tests, such as testing under different workloads (balanced, random, skewed) and at different workload skewness.
You can alter the tests by changing the code in `main` function of  `test_alltoall.cpp`.
You can write your own test program by following the existing code such as function `perftest_fixed_transfer_sz` in `test_alltoall.cpp`.
The performance results are printed and recorded in files.

When you change the number of servers involved in the testing, remember to change the `server_n` in the `main` function of `test_alltoall.cpp` and recompile the program.

# Python Interface Test
FLASH wraps the C++ scheduler into a python module, which can be called by AI framework such as Megatron-LM.
Run the following command to compile the Python interface and scheduler:
```
make clean
make flash
```
Make sure all the nodes compile the code of the same version.

To test whether FLASH Python interface is functioning correctly, alter the `MASTER_ADDR`, `NNODES`, `NODE_RANK` in each node's `torch_run.sh`, and launch test program with:
```
bash torch_run.sh
```
on each node.


# Dependency and Docker
Make sure the following libraries are successfully installed:
* ROCM
* Rccl
* Rccl-test
* MPI


There is a docker container that has already installed the above dependencies. You can launch a local helper script with:
```
bash <path-to-your-docker-script> v1.5
```
Or download the docker image as follows:
```
docker pull ghcr.io/a-dying-pig/msccl_docker/mscclpp_rocm:v1.5
# or
docker pull ghcr.io/microsoft/mscclpp/mscclpp:base-dev-rocm6.2
```
