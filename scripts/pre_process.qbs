#!/bin/bash
#PBS -A Local-LLM
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-node
#PBS -l select=1
#PBS -N preprocess-data
#PBS -j oe
#PBS -M sf850@scarletmail.rutgers.edu                                        
#PBS -m bae   
module add conda
conda activate env_conda
cd /lus/eagle/projects/Local-LLM/shuyuanfan/Megatron-LM/scripts
PYTHONUNBUFFERED=1 python /lus/eagle/projects/Local-LLM/shuyuanfan/Megatron-LM/tools/preprocess_data.py \
       --input /lus/eagle/projects/Local-LLM/shuyuanfan/data_prepare/dataset_downloaded.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --workers 32 \
       --append-eod
