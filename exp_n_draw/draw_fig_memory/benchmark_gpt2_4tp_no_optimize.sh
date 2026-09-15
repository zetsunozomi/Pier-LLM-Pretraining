#!/bin/bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4 
#SBATCH --output=gpt2_4gpu_no_outer_mem_opt.%j.out
#SBATCH --job-name=gpt2-4gpu-maxmem
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# nersc local file issue. disable user site packages
export PYTHONNOUSERSITE=1

# tepmorary cache directory for torchinductor and triton
export LOCAL_CACHE_DIR="/tmp/torch_compile_${SLURM_JOB_ID}"
# set torchinductor and triton cache directory
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_CACHE_DIR/triton"

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR

module add conda
conda activate diloco

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=4

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=29500
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/pscratch/sd/s/syfan/Pier/checkpoints
TENSORBOARD_LOGS_PATH=/pscratch/sd/s/syfan/Pier/logs
VOCAB_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-vocab.json
MERGE_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-merges.txt
DATA_PATH=/pscratch/sd/s/syfan/Pier/data/my-gpt2_text_document

GPT_MODEL_ARGS=(
    --num-layers 11
    --hidden-size 6144
    --ffn-hidden-size 26112
    --num-attention-heads 48
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 512
    --train-iters 60
    --weight-decay 0.1
    --adam-beta1 0.9 
    --adam-beta2 0.999
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 4e-4
    --lr-decay-style cosine 
    --min-lr 4e-5
    --lr-warmup-iters 50
    --lr-decay-iters 100000   
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 4
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
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --ckpt-format torch
)

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
srun torchrun -m torch.distributed.run --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
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
        --num-subgroup 1 \
        --momentum-warmup-steps 0\
        
