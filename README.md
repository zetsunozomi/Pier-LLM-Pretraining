# PIER: Efficient Large Language Model Pretraining with Relaxed Global Communication
[![arXiv](https://img.shields.io/badge/arXiv-2511.17849-b31b1b.svg)](https://arxiv.org/abs/2511.17849)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
## Introduction
This repository contains the implementation of PIER, a novel approach for low-communication Large Language Model (LLM) pretraining. It introduces relaxed global communication strategies to mitigate the overhead of synchronization in large-scale distributed training settings.

The codebase is built on top of [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM), inheriting its robust model parallelism features while introducing new optimizations for communication efficiency.

### Upstream credit

PIER is derived from NVIDIA Megatron-LM. Credit for the upstream code belongs to NVIDIA and the Megatron-LM contributors. The original copyright notices and third-party attributions are retained in the source files and [LICENSE](LICENSE).

## Usage
TODO: Put reproduction command here.

## Centered-reference development: first GPU gate

The new centered-reference method is under development. Its first cluster
handoff checks mixed-precision model commits, inner gradient normalization,
and the single-slot CUDA/NCCL reference protocol. It needs four GPUs and no
model weights or dataset. This is correctness validation, not a performance result.

See [the E0a runbook](experiments/centered_outer/README.md) for the sbatch command,
supported scope, output files, and current implementation limits.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{fan2025pier,
  title={Pier: Efficient Large Language Model pretraining with Relaxed Global Communication},
  author={Fan, Shuyuan and Zhang, Zhao},
  journal={arXiv preprint arXiv:2511.17849},
  year={2025}
}
```
