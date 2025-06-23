#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# === Configuration Parameters ===
motif_list='../../meme/motif_filtered.txt'  # CSV file containing TF names (first column)
check_folder='../Algorithm/DiscoveredMotifs/DeepSNR'  # Path to check for existing .jaspar files
model_name='DeepSNR'         # Options: DeepSNR or D_AEDNet


# Read TF names line by line (skip the CSV header)
tail -n +2 "$motif_list" | cut -d',' -f1 | while read tf_name; do
    # Trim whitespace
    tf_name=$(echo "$tf_name" | xargs)
    if [[ -z "$tf_name" ]]; then
        continue  # Skip empty lines
    fi

    jaspar_file="${check_folder}/${tf_name}/${tf_name}.jaspar"
    
    if [ -f "$jaspar_file" ]; then
        echo "⚠️  Motif file already exists. Skipping TF: $tf_name"
        continue
    fi

    echo "🚀 Processing TF: $tf_name using model: $model_name"
    
    python DeepLearning_Motif.py \
        --model "$model_name" \
        --tf "$tf_name" \
        --motif_len 11

    echo "✅ Finished TF: $tf_name"
    echo
done

echo "🎉 All TFs processed successfully!"
