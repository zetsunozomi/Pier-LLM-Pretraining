seq_length = 1024
global_batch_size = 512
tokens = seq_length * global_batch_size 
model_num_parameters = 1.56e9

#----------------------------------
num_gpu = 32
time_per_iteration =  3.44689475



FLOP_per_iteration = 6 * model_num_parameters  * tokens
# in second. from my log, 3377ms per iteration.


TFLOPS_per_gpu = ((FLOP_per_iteration / time_per_iteration)/num_gpu)/1e12
print(f"TFLOPS:{TFLOPS_per_gpu}")


print(f"mfu: {TFLOPS_per_gpu*100/130}%")
# A100 312 TFLOPS for bf16, 156 for fp32
# H200 1671? 