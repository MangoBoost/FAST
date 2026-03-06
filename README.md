# FAST

This repository contains the reference code for the paper:
**"FAST: An Efficient Scheduler for All-to-All GPU Communication"**.

Authors:
Yiran Lei, Dongjoo Lee, Liangyu Zhao, Daniar Kurniawan, Chanmyeong Kim, Heetaek Jeong, Changsu Kim, Hyeonseong Choi, Liangcheng Yu, Arvind Krishnamurthy, Justine Sherry, Eriko Nurvitadhi.

## Repository Layout

- `simulation/`: Algorithm-level simulator and benchmarking code for FAST scheduling ideas (matrix decomposition, scheduling policy studies, and plotting scripts).
- `nvidia/`: NVIDIA/NVSHMEM runtime implementation, including CUDA kernels, runtime scheduling, Python bindings, and multi-node test scripts.
- `amd/`: AMD/RCCL-oriented implementation and test harnesses, including corresponding build scripts and legacy experiment artifacts.

## License (Academic & Non-Commercial)

This repository is intended and licensed for academic and non-commercial research use only.
You are free to use, modify, and distribute this software for non-commercial, academic, and research purposes only.

Commercial use, including integration into commercial products or services, requires prior written permission from the authors.
For commercial licensing inquiries, please contact us at contact@mangoboost.io.

## Third-Party Code and Notices

This repository includes some source files derived from external open-source
projects.

- `nvidia/src/include/registration.h` is derived from the vLLM project
  (`vllm-project/vllm`), licensed under Apache License 2.0.

See `THIRD_PARTY_NOTICES.md` for details.