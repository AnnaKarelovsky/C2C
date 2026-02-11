#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python script/optimize/cache_optimize_training.py train \
  --dataset local/trajectory/qwen_instruct/qwen3_235b_instruct/hotpotqa/300_end/hotpotqa_dataset \
  --output-dir local/checkpoints/optCache_300end_full_no_thinking \
  --batch-size 1 --grad-accum 16 --lr 32e-3 \
  --wandb --wandb-name "optCache-300end-full-no-thinking-lr32e3-bs1ga16" \
  --no-thinking --max-length 4096