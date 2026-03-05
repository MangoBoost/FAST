# pyright: reportCallIssue=false

import os

import torch

_lib_path = os.path.join(os.path.dirname(__file__), "libflash.so")
torch.ops.load_library(_lib_path)
torch.classes.load_library(_lib_path)

flash_comm_t = torch.classes.my_classes.flash_comm_t

def get_nvshmem_init_id() -> torch.Tensor:
    return torch.ops.flash_nvshmem.get_nvshmem_init_id()