#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

motif_list='../../meme/motif_filtered.txt'
check_folder='../Algorithm/DiscoveredMotifs/BertSNR'

echo "🧪 Starting batch motif discovery task..."
# Read TF names from CSV (skip header)
tail -n +2 "$motif_list" | cut -d',' -f1 | while read tf; do
    jaspar_file="${check_folder}/${tf}/${tf}.jaspar"

    if [ -f "$jaspar_file" ]; then
        echo "⚠️  Motif file already exists. Skipping TF: $tf"
        continue
    fi

    echo "🔍 Processing TF: $tf"
    python GenerateMotif.py --motif_name "$tf"
    echo "✅ Finished TF: $tf"
done

echo "🎉 All motif discovery tasks completed!"
