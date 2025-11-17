import os
import sys
import torch
print("Python executable:", sys.executable)
import einops
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"This is global rank {rank}, local rank {local_rank}, world size {world_size}")
