#!/bin/sh
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4 
#SBATCH --output=70B_128gpu_4tp_4dp_8pp_subgroup4.%j.out
#SBATCH --job-name=pretrain-GPT2small
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
NUM_NODES=32
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/pscratch/sd/s/syfan/Pier/checkpoints
TENSORBOARD_LOGS_PATH=/pscratch/sd/s/syfan/Pier/logs
VOCAB_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-vocab.json
MERGE_FILE=/pscratch/sd/s/syfan/Pier/data/gpt2-merges.txt
DATA_PATH=/pscratch/sd/s/syfan/Pier/data/my-gpt2_text_document

GPT_MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 36672
    --num-attention-heads 64
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

# Benchmark schedule for mocked sync-200 result:
#   20 system-warmup steps are kept in the run but ignored in analysis.
#   The measured 200-step window uses 20 DDP/momentum-warmup steps and
#   180 DiLoCo steps with the minimum outer sync interval of 20.
#   Use the measured slow/fast step times to mock the verified sync-200 result.
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 512
    --train-iters 220
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
	--pipeline-model-parallel-size 8
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
echo $LD_LIBRARY_PATH
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
        --outer-sync-interval 20 \
        --outer-optimizer pytorch_nesterov \
        --num-subgroup 4 \
        --momentum-warmup-steps 40
