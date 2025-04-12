import torch
import torch.distributed as dist
import os

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # 确保每个进程绑定到不同的 GPU
    print(f"Rank {local_rank} using GPU {torch.cuda.current_device()}")

    # 确保总共有 4 个进程（rank0, rank1, rank2, rank3）
    assert world_size == 4, "This example requires exactly 4 ranks."

    # 定义子组：rank0 和 rank1 一组，rank2 和 rank3 一组
    subgroup_ranks_0_1 = [0, 1]
    subgroup_ranks_2_3 = [2, 3]

    subgroup_0_1 = dist.new_group(ranks=subgroup_ranks_0_1)
    subgroup_2_3 = dist.new_group(ranks=subgroup_ranks_2_3)

    if rank in [0,1]:
        subgroup = subgroup_0_1
    if rank in [2,3]:
        subgroup = subgroup_2_3
    # 每个 rank 初始化一个张量
    tensor = torch.tensor([rank], dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor([rank], dtype=torch.float32)

    print(f"Rank {rank}: Before allreduce, tensor = {tensor}")

    # 在子组内执行 allreduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=subgroup)

    print(f"Rank {rank}: After allreduce, tensor = {tensor}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()