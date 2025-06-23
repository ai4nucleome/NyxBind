#!/bin/bash

#path for sequence data
export DATA_PATH=./33JASPAR
#path for attenscore
export PREDICTION_ROOT=./attention_output/NyxBind
#path for output motif
export RESULT_ROOT=./result/NyxBind

for SUBDIR in "$DATA_PATH"/*/; do
    SUBDIR=${SUBDIR%/}
    BASENAME=$(basename "$SUBDIR")
    PREDICTION_PATH="$PREDICTION_ROOT/$BASENAME"
    MOTIF_PATH="$RESULT_ROOT/$BASENAME"

    echo "Processing: $BASENAME"
    echo "DATA_DIR=$SUBDIR"
    echo "PREDICTION_PATH=$PREDICTION_PATH"
    echo "MOTIF_PATH=$MOTIF_PATH"

    mkdir -p "$MOTIF_PATH"

    python find_motifs.py \
        --data_dir "$SUBDIR" \
        --predict_dir "$PREDICTION_PATH" \
        --window_size 11 \
        --min_len 6 \
        --top_k 1 \
        --pval_cutoff 0.005 \
        --min_n_motif 10 \
        --align_all_ties \
        --save_file_dir "$MOTIF_PATH" \
        --verbose
    echo "🐾🐶Finish analysis $BASENAME"
done