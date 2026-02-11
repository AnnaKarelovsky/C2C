#!/bin/bash
CUDA_VISIBLE_DEVICES=8 python script/optimize/standard_training.py train \
  --dataset local/trajectory/qwen_instruct/qwen3_235b_instruct/hotpotqa/300_end/hotpotqa_dataset \
  --output-dir local/checkpoints/sft_300end_full_no_thinking \
  --batch-size 1 --grad-accum 16 --lr 16e-5 \
  --wandb --wandb-name "sft-300end-full-no-thinking-lr16e5-bs1ga8" \
  --no-thinking --max-length 4096
