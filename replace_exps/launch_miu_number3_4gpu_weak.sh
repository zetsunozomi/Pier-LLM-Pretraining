#!/bin/sh
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=36:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=miu_exp4_weak4.%j.out
#SBATCH --job-name=pretrain-GPT2small
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module add conda
conda activate diloco

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=4

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/pscratch/sd/s/syfan/Pier/checkpoints_exp4_weak4
TENSORBOARD_LOGS_PATH=/pscratch/sd/s/syfan/Pier/logs
VOCAB_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-vocab.json
MERGE_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-merges.txt
DATA_PATH=/pscratch/sd/s/syfan/Pier/data/my-gpt2_text_document

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 256
    --train-iters 200000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.999
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --lr 2e-4
    --lr-decay-style cosine
    --min-lr 2e-5
    --lr-warmup-iters 4000
    --lr-decay-iters 200000
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 5000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --ckpt-format torch
)

srun python3 -m torch.distributed.run --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        /pscratch/sd/s/syfan/Pier/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]}\
        --transformer-impl local \
        --use-legacy-models \
        --use-flash-attn \
        --outer-sync-interval 50 \
        --outer-optimizer pytorch_nesterov \
        --num-subgroup 4 \
        --momentum-warmup-steps 20000
