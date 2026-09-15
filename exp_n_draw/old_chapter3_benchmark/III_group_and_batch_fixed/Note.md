fixed subgroup, fixed batch size, add number of gpus.

run small model:
subgroup == 8.

8gpus, 16gpus, 32gpus, microbatchsize == 16, accumulation == 512.

ddp：
8 686.1443298969073
16 438.533
32 297.801

diloco:
8 517.602
16 277.281
32 154.25
-----------------------------
run medium model:

32gpus, 64gpus, 128gpus, micro == 4, accumulation == 512.

(extract from 500 to 1500)
diloco:
32: 467.289
64: 324.879
128: 220.018

ddp: 
32: 969.159
64: 812.394
128: 620.094

-----------------------------
run xl model: extract: 500-1000

diloco (fixed subgroup 64):
64 1746.94
128 1267.894
256 919.218

ddp:
4gpu 21726.108
8gpu 13282.264
16gpu 8030.34
32gpu 5374.374
64gpu  3961.222
128gpu 3255.924
256gpu 2961.51

FGCS