# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from packaging.version import Version
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .finalize_model_grads import finalize_model_grads
from .torch_fully_sharded_data_parallel import TorchFullyShardedDataParallel

if DistributedDataParallelConfig.low_comm_mode:
    from .my_distributed_data_parallel import MyDistributedDataParallel as DistributedDataParallel
else:
    from .distributed_data_parallel import DistributedDataParallel
