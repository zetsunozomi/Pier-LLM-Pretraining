#!/bin/bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=downstream_eval.%j.out
#SBATCH --job-name=downstream-eval
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

unset LD_LIBRARY_PATH
export NVIDIA_PYTORCH_VERSION="invalid"

path="/pscratch/sd/s/syfan/Pier/checkpoints_pier_small/iter_0100000/mp_rank_00"
MODEL_ARGS="pretrained=${path},dtype=float"
OUT_FILE='pier_small_eval'

{
  echo "--- Starting SuperGLUE ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks super-glue-lm-eval-v1 --batch_size 8

  echo "--- Starting Lambada OpenAI ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks lambada_openai --batch_size 8

  echo "--- Starting Lambada Standard ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks lambada_standard --batch_size 8

  echo "--- Starting RACE ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks race --batch_size 8

  echo "--- Starting MathQA ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks mathqa --batch_size 8 --trust_remote_code

  echo "--- Starting PIQA ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks piqa --batch_size 8

  echo "--- Starting Winogrande ---"
  lm_eval --model hf --model_args "$MODEL_ARGS" --tasks winogrande --batch_size 8

} 2>&1 | tee "${OUT_FILE}.log"
