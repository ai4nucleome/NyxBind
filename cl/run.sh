#!/bin/bash
# This script is used to train NyxBind with multiple TFBS datasets using contrastive learning.

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False
export WANDB_DISABLED=False

#data
BASE_PATH="../data/159chipseq"

#pretrain dnabert2
BASE_MODEL="./model/DNABERT-2-117M"

python cl.py \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --num_epochs 3 \
  --max_seq_length 30 \
  --random_seed 42 \
  --learning_rate 3e-5 \
  --base_path $BASE_PATH \
  --model_name_or_path $BASE_MODEL \
  --train_numbers 10000 \
  --test_numbers 1000 \
  --start_layer 11\
  --model_save_root "output/NyxBind"