#!/bin/bash
m=$1

# Select model
if [ "$m" -eq 0 ]; then
    model=InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    run_name=500M_multi_species_v2
elif [ "$m" -eq 1 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-human-ref
    run_name=NT_500_human
elif [ "$m" -eq 2 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-1000g
    run_name=NT_2500_1000g
elif [ "$m" -eq 3 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    run_name=NT_2500_multi
else
    echo "❌ Invalid model selection argument."
    exit 1
fi

echo "🔧 Selected model: $model"

data_path=../../data/159chipseq
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=False
export WANDB_DISABLED=True
lr=1e-4
savename=${run_name}-${lr}

for seed in 42
do
    for data in "$data_path"/*/
    do 
        folder_name=$(basename "$data")
        result_file="output/$savename/results/NT_${lr}_${folder_name}_${seed}/eval_results.json"

        # Skip if result already exists
        if [ -f "$result_file" ]; then
            echo "[SKIP] $folder_name already completed. Skipping..."
            continue
        fi

        echo "[RUNNING] Training on $folder_name..."
        python train.py \
            --model_name_or_path ${model} \
            --data_path  "$data" \
            --kmer -1 \
            --run_name NT_${lr}_${folder_name}_${seed} \
            --model_max_length 30 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16  \
            --save_steps 200 \
            --output_dir output/$savename \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done
