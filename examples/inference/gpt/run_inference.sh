module add conda
conda activate megatron-lm
export PYTHONPATH=/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco 
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NUM_NODES=1
GPUS_PER_NODE=1

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

python /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/examples/inference/gpt/gpt_static_inference.py \
    ${GPT_MODEL_ARGS[@]} \
    --load /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/gpt2-small-logs/exp6/iter_0100000 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/data/gpt2-vocab.json \
    --merge-file /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/data/gpt2-merges.txt \
    --prompts "Hello, world!" \
    --temperature 1.0 \
    --top_k 50 \
    --top_p 0.9 \
    --num-tokens-to-generate 32
