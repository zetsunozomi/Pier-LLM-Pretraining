#!/bin/bash
#PBS -A Local-LLM
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-node
#PBS -l select=1
#PBS -N pretrain-GPT2small
#PBS -j oe
#PBS -M sf850@scarletmail.rutgers.edu                                        
#PBS -m bae 

export NCCL_NET=Socket
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

module add conda
conda activate megatron-lm
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/checkpoints
TENSORBOARD_LOGS_PATH=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/logs
VOCAB_FILE=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/data/gpt2-vocab.json
MERGE_FILE=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/data/gpt2-merges.txt
DATA_PATH=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/data/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

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
    --global-batch-size 512
    --train-iters 100000  
    --weight-decay 0.1
    --adam-beta1 0.9 
    --adam-beta2 0.999
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 4e-4
    --lr-decay-style cosine 
    --min-lr 4e-5
    --lr-warmup-iters 2000
    --lr-decay-iters 100000   
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
    --save-interval 4000
    --eval-interval 1000
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --ckpt-format torch
)
# --load-iteration 1000
torchrun ${DISTRIBUTED_ARGS[@]} /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}\
    --transformer-impl local \
    --outer-sync-interval 50 \
    --outer-optimizer lars \
    --use-legacy-models

# Using subgroup number == 8
# outer learning rate using warmup.
# Double training steps.
# weight decay 0.1
# lr 4e-4 
# batchsize 16/512