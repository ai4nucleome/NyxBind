#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False
export WANDB_DISABLED=True

data_path=../../data/1
lr=3e-5

#BERT-TFBS_N: using NyxBind as basemodel
#BERT-TFBS: using DNABERT2 as basemodel
model_path='../../cl/output/NyxBind'


savename=BERTTFBS_${lr}

for seed in 42
do    
    for data in "$data_path"/*/
    do 
        folder_name=$(basename "$data")  
        python train.py \
            --model_name_or_path $model_path \
            --data_path  $data \
            --kmer -1 \
            --run_name bfcl_${lr}_${folder_name}_${seed} \
            --model_max_length 30 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 15 \
            --fp16 \
            --save_steps 100 \
            --output_dir output/$savename \
            --evaluation_strategy steps \
            --eval_steps 100 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done
done