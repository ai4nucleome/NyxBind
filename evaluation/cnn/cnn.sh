#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False
export WANDB_DISABLED=True

data_path=../../data/159chipseq

lr=1e-3
model_name='DeepBind'
#DeepBind OR DanQ

savename=${model_name}-lr-${lr}-42
for seed in 42
do    
    for data in "$data_path"/*/
    do 
        folder_name=$(basename "$data") 
        python cnn.py \
            --model_name_or_path $model_name  \
            --data_path  $data \
            --run_name ${model_name}_${lr}_${folder_name}_${seed} \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 15 \
            --fp16 False \
            --save_steps 50 \
            --output_dir output/$savename \
            --evaluation_strategy steps \
            --eval_steps 50 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info 
    done
done