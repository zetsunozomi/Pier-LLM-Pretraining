# From table 1
A100_peak = 137/0.44

sequence_length = 1024
hidden_size = 1600
number_layers = 48
batch_size = 512
vocabulary_size = 50257
term_2 = sequence_length/(6*hidden_size)
term_3 = vocabulary_size /(16*number_layers*hidden_size)
flops = 96 * batch_size * sequence_length * number_layers * hidden_size*hidden_size*(1+term_2+term_3)
Tflops = flops/1e12
Tflops = 5213372232499200/1e12
print(f"Total TFLOPs per step: {Tflops}")

num_gpus = 1
iteration = 5.711935
per_GPU_TFLOP = Tflops/(num_gpus*iteration)
print(f"per GPU TFLOP/s :{per_GPU_TFLOP}")
print(f"MFU : {per_GPU_TFLOP/A100_peak}")
