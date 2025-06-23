#!/bin/bash
# Filename: run_tomtom_testmotif_vs_human.sh
# Function: Merge all MEME files from testmotif and compare them with the pre-merged human.meme using Tomtom

# Set paths
pfm_file="./human/human.meme"  # ✅ already merged
title='NyxBind'
testmotif_dir="./$title"
merged_dir="./merged_meme/$title"
tomtom_results_dir="./tomtom_results_merged/$title"

# Create output directories
mkdir -p "$merged_dir"
mkdir -p "$tomtom_results_dir"

# Merge MEME files from testmotif
merged_testmotif="${merged_dir}/merged_testmotif.meme"

echo "🔧 Merging all MEME files from $testmotif_dir into $merged_testmotif"
first_file=$(find "$testmotif_dir" -name "*.meme" | head -n 1)
if [[ -z "$first_file" ]]; then
    echo "❌ No MEME files found. Exiting."
    exit 1
fi

# Extract header (everything before the first MOTIF line)
awk '/^MOTIF/ {exit} {print}' "$first_file" > "$merged_testmotif"

# Append all MOTIF entries
for f in "$testmotif_dir"/*.meme; do
    echo "  -> Adding $f"
    awk 'flag {print} /^MOTIF/ {flag=1; print}' "$f" >> "$merged_testmotif"
    echo "" >> "$merged_testmotif"
done

echo "✅ Merge complete: $merged_testmotif"

# Run Tomtom comparison
echo "Running Tomtom comparison:"
echo "  Query : $merged_testmotif"
echo "  Target: $pfm_file"

tomtom -no-ssc -oc "$tomtom_results_dir" -evalue -thresh 0.5 "$merged_testmotif" "$pfm_file"

if [[ -f "${tomtom_results_dir}/tomtom.tsv" ]]; then
    echo "✅ Tomtom comparison complete. Results saved in $tomtom_results_dir"
else
    echo "❌ Tomtom comparison failed or no results were generated."
fi

python filter.py \
  --name $title \
  --id_file ./motif_filtered.txt \
  --tom_result $tomtom_results_dir/tomtom.tsv \
  --output_dir ./filter_res