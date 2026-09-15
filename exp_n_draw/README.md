exp1: repeat sophia exp9, but parallel subgroup is 4.

exp2: run subgroup == 16 fron scratch on this machine, same as polaris exp 5

exp3: subgroup16: no lazy start, weak scale

exp4: same as exp3 but lazy start, work as subgroup 16 with lazy start, weak scale

exp5: subgroup == 8, from scratch, gpt345 Medium training.

exp6: subgroup 32, all other hyper-parameters same *Here is the warmup only*

exp7: subgroup 32, miu==0.95,

exp8: from scratch

exp9：miu==0.95 outer lr==1.5 explode

exp10: outer momentum lr. it doesnt make sense.

exp11: warmup miu from 0.5 to 0.9, lr 1.2

maybe miu 0.99

exp12: quite complicated hyper-parameters tuning.

subgroup:32
--lr 3e-4
--min-lr 3e-5

outer tuning:
warmup miu:
    miu_min = 0.5
    miu_max = 0.95

    miu_warmup_steps=5000
    if iteration - start_iteration < miu_warmup_steps:
        miu = miu_min + (miu_max - miu_min)*(iteration - start_iteration)/miu_warmup_steps
    else:
        miu = miu_max
warmup outer lr to 1.1, and use 1 for last 20k steps.
    lr_warmup_steps = 2000
    if iteration > 80000:
        base_lr = 1
    else:
        base_lr = 1.1
    if iteration - start_iteration < lr_warmup_steps:
        outer_lr = base_lr * (iteration - start_iteration)/lr_warmup_steps
    else:
        outer_lr = base_lr
    if rank == 0:
        print(f"lr = {outer_lr}")

exp13: same as exp 12, but lr warmup at 6000

exp14(running): same as exp13 but from scratch diloco

exp15: large miu, complicated tuning.
            elif args.outer_optimizer=="pytorch_nesterov":
                # code for large miu start.
                if iteration < 15000:
                    miu = 0.99
                elif iteration < 20000:
                    miu = 0.95
                else: 
                    miu = 0.9
                lr_warmup_steps = 10000
                # determine the base_lr automatically?
                if iteration > 80000:
                    base_lr = 0.9
                elif iteration < 20000:
                    base_lr = 1
                else:
                    base_lr = 1.1
                if iteration - start_iteration < lr_warmup_steps:
                    outer_lr = base_lr * (iteration - start_iteration)/lr_warmup_steps
                else:
                    outer_lr = base_lr