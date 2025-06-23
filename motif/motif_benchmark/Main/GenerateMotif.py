import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio.Seq import Seq
from Bio import motifs

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Dataset.DataLoader
import Dataset.DataReader
import Model.BertSNR
import Utils.Metrics
import Utils.Threshold
from Utils.MotifDiscovery import MotifDiscovery


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def kmer2seq(kmers):
    """Convert k-mer tokens back to the original DNA sequence"""
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


if __name__ == "__main__":
    print("🧬 BertSNR Motif Discovery Script Starting...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_name", type=str, required=True, help="Motif/TF name (e.g., CDX2)")
    args = parser.parse_args()

    TFsName = args.motif_name
    KMER = 3
    motif_len = 11
    threshold_value = 0.5
    batch_size = 64

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"📍 TF name: {TFsName}, Using GPU: {use_gpu}")

    set_seed(1)

    # [1] Load test sequences
    print("🔍 [1/6] Loading test sequence data...")
    data_path = f'../Dataset/ReMap/{TFsName}.bed.txt'
    test_sequences = Dataset.DataReader.DataReaderPrecitBERT(data_path)
    test_sequences = np.array(test_sequences)
    print(f"✅ Loaded. Total sequences: {len(test_sequences)}")

    # [2] Load model and weights
    print("🔧 [2/6] Initializing model and loading weights...")
    model = Model.BertSNR.BERTSNR(KMER).to(device)
    weight_path = f'ModelWeight/multiModel/{TFsName}/pytorch_model.bin'
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # [3] Build DataLoader
    print("📦 [3/6] Building data loader...")
    test_loader = Dataset.DataLoader.SampleLoaderPredictUnlabelBERT(
        Sequence=test_sequences, BatchSize=batch_size)
    print(f"✅ DataLoader built. Total batches: {len(test_loader)}")

    # [4] Inference
    print("⚙️ [4/6] Running model inference...")
    preds = []
    for step, batch in enumerate(tqdm(test_loader, desc="Inferencing")):
        with torch.no_grad():
            _, token_logits = model(batch)
            prediction = Utils.Threshold.Threshold(token_logits.cpu(), threshold_value)
            preds.append(prediction)

    predictions = np.concatenate(preds)
    labels = predictions.reshape(-1, 98).astype(np.int64)

    # [5] Recover sequences and extract motifs
    print("🧩 [5/6] Extracting motif segments...")
    sequence_list = [kmer2seq(row) for row in test_sequences]
    output_dir = f'../Algorithm/DiscoveredMotifs/BertSNR/{TFsName}'
    os.makedirs(output_dir, exist_ok=True)

    motif_file = os.path.join(output_dir, f'{TFsName}.txt')
    seqs = []

    with open(motif_file, 'w') as f:
        for i, row in enumerate(labels):
            dense_label = [0] * (len(row) + KMER - 1)
            for j in range(len(row)):
                if row[j] == 1:
                    for k in range(KMER):
                        dense_label[j + k] = 1
            motif = MotifDiscovery(sequence_list[i], dense_label, motif_len)
            if 'N' not in motif and len(motif) == motif_len:
                seqs.append(motif)
                f.write(motif + '\n')

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} sequences, valid motifs so far: {len(seqs)}")

    print(f"✅ Motif extraction complete. Total valid motifs: {len(seqs)}")

    # [6] Generate PWM and WebLogo
    if seqs:
        print("📊 [6/6] Generating PWM and WebLogo...")
        m = motifs.create([Seq(s) for s in seqs])
        m.pseudocounts = 0.01
        m.background = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}

        jaspar_file = os.path.join(output_dir, f"{TFsName}.jaspar")
        with open(jaspar_file, 'w') as f:
            f.write(m.format("jaspar"))
        print(f"✅ JASPAR PWM saved to: {jaspar_file}")

        weblogo_file = os.path.join(output_dir, f"{TFsName}_weblogo.png")
        m.weblogo(weblogo_file,
                  format='png_print',
                  show_fineprint=False,
                  show_ends=False,
                  color_scheme='color_classic')
        print(f"✅ WebLogo image saved to: {weblogo_file}")
    else:
        print("⚠️ No valid motifs found. Skipping WebLogo generation.")
