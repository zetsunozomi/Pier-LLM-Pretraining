module add conda
conda activate megatron-lm
PYTHONPATH=/lus/eagle/projects/Local-LLM/shuyuanfan/Megatron-LM python3 convert.py --model-type GPT \
    --load-dir /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/checkpoints/iter_0000050/mp_rank_00/model_optim_rng.pt \
    --save-dir /lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/checkpoint_converted \
    --loader megatron \
    --saver mcore
    