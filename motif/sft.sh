#!/bin/bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU 0

#path for sequence data
DATA_ROOT=${1:-"./33JASPAR"} 
#path to FT model weights folder
MODEL_ROOT='../finetune/ft/output/NyxBind-33-ft/weights' 
OUTPUT_ROOT="./attention_output/NyxBind"

for SUBDIR in "$DATA_ROOT"/*/; do
    SUBDIR=${SUBDIR%/}
    BASENAME=$(basename "$SUBDIR")
    MODEL_PATH="$MODEL_ROOT/$BASENAME"
    ATTEN_FILE="$OUTPUT_ROOT/$BASENAME/atten.npy"

    if [ ! -f "$SUBDIR/motif.csv" ]; then
        echo "⚠️ Skipping $SUBDIR, motif.csv not found"
        continue
    fi
    if [ -f "$ATTEN_FILE" ]; then
        echo "✅ Skipping $SUBDIR, $ATTEN_FILE already exists"
        continue
    fi

    echo "🙂 Starting process for $MODEL_PATH"

    python score_from_sft.py \
        --model_path "$MODEL_PATH" \
        --data_path "$SUBDIR" \
        --output_root "$OUTPUT_ROOT" \
        --selected_layers 11 \
        --batch_size 256 \
        --max_length 30

    echo "✅ Finished processing $SUBDIR"
done
