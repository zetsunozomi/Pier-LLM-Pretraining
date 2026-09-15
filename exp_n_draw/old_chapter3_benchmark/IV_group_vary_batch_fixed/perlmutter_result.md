**perlmutter. small model. vary subgroup, add gpus.**

microbatchsize == 16, accumulation == 512.
X subgroups, X gpus:

DiLoCo
X==2 1997.655
X==4 1011.695
X==8 517.602 (the same as 3.1)
X==16 273.469
X==32 151.661

DDP(same as experiment in 3.1)

8gpus, 16gpus, 32gpus, microbatchsize == 16, accumulation == 512.

ddp：
1 3984.3360000000002
2 2014.364
4 952.794
8 686.1443298969073
16 438.533
32 297.801

**perlmutter. Medium model. vary subgroup, add gpus.**

ddp：
1 11351.476
2 5912.502
4 3126.64
8 2072.53
16 1364.296 (above is added for 4.1 experimen)
32 969.159 (below is the same as 3.1 ddp medium)
64 812.394
128 620.094

diloco:
1 11351.476 (the same as ddp)
2 5889.432
4 3057.892
8 1602.692
16 842.425
32: 467.289 (the same as 3.1 medium)
64: 292.383
128: 209.787

**perlmutter. XL model. vary subgroup, add gpus.**

ddp:
1gpu 78489.28666666667
2gpu 40337.29
4gpu 21629.8
8gpu 13237.493333333334
16gpu 8030.34
32gpu 5363.776666666667
64gpu  3961.222 (from 3) 3981.4066666666668 (200 to 500)
128gpu 3255.924 (from 3) 3268.8566666666666 (200 to 500)

diloco:
2 40376.486666666664
4 21452.093333333334
8 11236.143333333333
16 5876.836666666667
32 3073.2433333333333
64 1741.4866666666667
128 1120.77