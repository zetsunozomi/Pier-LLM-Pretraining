# LM Evaluation Harness Setup

Run these commands yourself when you are ready to install the harness into a new conda environment:

```bash
module load conda
conda create -n lm_eval_harness python=3.11 pip -y
conda activate lm_eval_harness

cd /pscratch/sd/s/syfan/lm-evaluation-harness
pip install -e ".[hf]"
```

Notes:

- The local checkout requires Python `>=3.10`.
- The base package does not install model backends. The `hf` extra installs Transformers, PyTorch, Accelerate, and PEFT.
- Install vLLM only if you plan to use the vLLM backend:

```bash
cd /pscratch/sd/s/syfan/lm-evaluation-harness
pip install -e ".[vllm]"
```

After installation, submit the entry script from this directory:

```bash
cd /pscratch/sd/s/syfan/Pier/downstream_task_workpath
sbatch run_single.sh /path/to/hf_model_or_checkpoint
```

Common overrides:

```bash
MODEL_PATH=/path/to/hf_model_or_checkpoint \
BATCH_SIZE=8 \
sbatch run_single.sh
```
