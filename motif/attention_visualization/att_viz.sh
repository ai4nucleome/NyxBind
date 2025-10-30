#!/bin/bash
# Example script for visualizing attention patterns from DNA sequence models
# Compare attention between base model (DNABERT-2) and fine-tuned model (NyxBind)

# Example DNA sequence (100bp)
SEQ=CTGGTGTGCTGTAGCCATGTCTTAGGCAAGTCCCCTTGTGCATGTTCCTTATCTGTGCCTGCAGGCTGTTCTTTTGTTTGAAAGGATTCATCTGAGCACCC
# Transcription factor name
TF=GATA2

# Path to your base model (DNABERT-2)
BASE_MODEL_PATH="./model/DNABERT-2-117M"
# Path to your fine-tuned model (NyxBind)
FINETUNED_MODEL_PATH='../../cl/output/NyxBind'

# ===== Extract and visualize attention from base model =====
echo "Processing base model (DNABERT-2)..."
python extract.py \
    --model_path $BASE_MODEL_PATH \
    --sequence $SEQ \
    --output_dir ./attention_vis \
    --save_name $TF-base

python visualize_attention.py \
    --npy_path ./attention_vis/$TF-base.npy \
    --model_path $BASE_MODEL_PATH \
    --sequence $SEQ \
    --save_path ./output/$TF-base.svg

# ===== Extract and visualize attention from fine-tuned model =====
echo "Processing fine-tuned model (NyxBind)..."
python extract.py \
    --model_path $FINETUNED_MODEL_PATH \
    --sequence $SEQ \
    --output_dir ./attention_vis \
    --save_name $TF-finetuned

python visualize_attention.py \
    --npy_path ./attention_vis/$TF-finetuned.npy \
    --model_path $FINETUNED_MODEL_PATH \
    --sequence $SEQ \
    --save_path ./output/$TF-finetuned.svg

echo "Done! Attention visualizations saved to ./output/"

