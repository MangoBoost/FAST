
import time
import torch
import torch.distributed as dist
from flash_utils import flash_comm_t, get_nvshmem_init_id
import os

@torch.no_grad()
def test():
    assert os.environ["NVSHMEM_IB_ENABLE_IBGDA"]
    torch.distributed.init_process_group(
        backend="cpu:gloo"
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    dev_n = torch.cuda.device_count()
    if rank == 0:
        uid = get_nvshmem_init_id()
    else:
        uid = torch.zeros((128,), dtype=torch.uint8, device="cpu")
    dist.broadcast(uid, src=0)
    torch.cuda.set_device(local_rank)
    assert rank % dev_n == local_rank
    flash_comm = flash_comm_t(rank, dev_n, world_size, uid)
    # flash_comm.test_basic_alltoall()
    flash_comm.test_flash_alltoall()
    # for power in range(24, 25):
    #     for test_id in range(0, 1):
    #         torch.distributed.barrier() 
    #         flash_comm.test_flash_under_different_transfer_sz(power, test_id)
    dist.destroy_process_group()

if __name__ == "__main__":
    test()
