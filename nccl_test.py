# test_nccl_simple.py
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

rank = dist.get_rank()
t = torch.tensor([1.0], device=f"cuda:{rank}")
dist.all_reduce(t)
t /= dist.get_world_size()

print(f"[RANK {rank}] tensor: {t.item()}")
