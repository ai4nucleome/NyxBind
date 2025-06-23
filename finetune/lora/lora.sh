#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False
export WANDB_DISABLED=True

data_path='../../data/159chipseq'

#'../../model/DNABERT2-117M' for finetune DNABERT2
model_path='../../cl/output/NyxBind'
lr=7e-4

for seed in 42
do
    for data in "$data_path"/*/
    do 
        folder_name=$(basename "$data")
        result_file="output/$savename/results/LoRA_${lr}_${folder_name}_${seed}/eval_results.json"
        if [ -f "$result_file" ]; then
            echo "[SKIP] $folder_name already completed, skipping."
            continue
        fi
        echo "[RUNNING] Starting training for $folder_name..."
        python train.py \
            --model_name_or_path $model_path \
            --data_path  $data \
            --kmer -1 \
            --run_name LoRA_${lr}_${folder_name}_${seed} \
            --model_max_length 30 \
            --use_lora \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 100 \
            --output_dir output/NyxBind-LoRA${lr} \
            --evaluation_strategy steps \
            --eval_steps 100 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done
