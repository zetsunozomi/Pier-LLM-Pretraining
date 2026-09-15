#!/bin/bash
echo "for Pier"
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Pier/7B_64gpu_4tp_4dp_4pp_subgroup4.47469255.out
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Pier/7B_128gpu_4tp_8dp_4pp_subgroup8.47533215.out
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Pier/7B_256gpu_4tp_16dp_4pp_subgroup16.47632703.out

echo "for Megatron"
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Megatron-LM/7B_64gpu_4tp_4dp_4pp.47469295.out
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Megatron-LM/7B_128gpu_4tp_8dp_4pp.47533246.out
python get_benchmark_result.py --path /pscratch/sd/s/syfan/Megatron-LM/7B_256gpu_4tp_16dp_4pp.47632716.out