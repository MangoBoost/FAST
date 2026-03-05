import torch
import FlashAllToAll
import numpy as np
import os
from datetime import timedelta
import time

torch_dtype_map = {
    torch.float32: 7,
    torch.float64: 8,
    torch.float16: 6,
    torch.bfloat16: 9,
    torch.uint8: 1,
    torch.uint32: 3,
    torch.uint64: 5,
    torch.int8: 0,
    torch.int32: 2,
    torch.int64: 4
}

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.environ["MASTER_ADDR"]
device_count = torch.cuda.device_count()
torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)
device_id = torch.device(f'cuda:{local_rank}')

if rank == 0:
    server_store = torch.distributed.TCPStore(master_addr, 31000, world_size, True, timedelta(seconds=30))
else:
    client_store = torch.distributed.TCPStore(master_addr, 31000, world_size, False)

if rank == 0:
    id_str = torch.cuda.nccl.unique_id()
    server_store.set("commID", id_str)
    id_str2 = server_store.get("commID")
else:
    id_str = client_store.get("commID")
torch.distributed.barrier()
embedding_sz = 4096
dtype = torch.float32
flash = FlashAllToAll.flash_t(rank, world_size, world_size // device_count, device_count, embedding_sz, torch_dtype_map[dtype], id_str)
if rank == 0:
    workload = np.random.randint(1024, size=world_size * world_size, dtype=np.int32)
    for i in range(world_size):
        workload[i * world_size + i] = 0
    workload_str = workload.tobytes()
    server_store.set("workload", workload_str)
else:
    workload_str = client_store.get("workload")
    workload = np.frombuffer(workload_str, dtype=np.int32)

input_array = np.empty((0, embedding_sz))
input_splits = []
output_splits = []
for dst in range(world_size):
    input_array = np.append(input_array, np.random.rand(workload[rank * world_size + dst], embedding_sz), axis = 0)
    input_splits.append(workload[rank * world_size + dst])
    output_splits.append(workload[dst * world_size + rank])
output_sz = sum(output_splits) * embedding_sz

input_tensor = torch.from_numpy(input_array).float().contiguous().to(device="cuda")
# if rank == 0:
#     print(input_array)
#     print(input_tensor.to(device="cpu").numpy())
#     print(input_tensor.size())
#     flash.print_tensor(input_tensor, 1)
#     print(input_splits)
#     print(output_splits)
#     print([sum(output_splits)] + list(input_tensor.size()[1:]))
output_tensor = torch.zeros(
    size=[sum(output_splits)] + list(input_tensor.size()[1:]),
    dtype=input_tensor.dtype,
    device=torch.cuda.current_device(),
).contiguous()
output_tensor_verify = torch.zeros(
    size=[sum(output_splits)] + list(input_tensor.size()[1:]),
    dtype=input_tensor.dtype,
    device=torch.cuda.current_device(),
).contiguous()

send_tensor = torch.zeros(size=[204800, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
recv_tensor = torch.zeros(size=[204800, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
lbsend_tensor = torch.zeros(size=[2048, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
lbrecv_tensor = torch.zeros(size=[2048, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
cros1_tensor = torch.zeros(size=[204800, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
cros2_tensor = torch.zeros(size=[204800, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()
rstr_tensor = torch.zeros(size=[102400, embedding_sz],dtype=dtype,device=f"cuda:{local_rank}").contiguous()

torch.distributed.barrier()
t1 = time.time()
flash.schedule(workload)
flash.all_to_all_2(input_tensor, output_tensor, send_tensor, recv_tensor, lbsend_tensor,lbrecv_tensor,cros1_tensor,cros2_tensor,rstr_tensor)
t2 = time.time()

torch.distributed.barrier()
t3 = time.time()
torch.distributed.all_to_all_single(output_tensor_verify, input_tensor, output_splits, input_splits)
t4 = time.time()

if rank == 0:
    print("flash alltoall:")
    print(output_tensor.cpu().numpy())
    print("torch alltoall:")
    print(output_tensor_verify.cpu().numpy())
    print("result equal:")
    print(np.array_equal(output_tensor.cpu().numpy(), output_tensor_verify.cpu().numpy()))
    print(f"flash time: {t2 - t1} s, torch time: {t4 - t3} s")