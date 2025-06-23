import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import tqdm

from Bio.Seq import Seq
from Bio import motifs

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Motif Discovery")
parser.add_argument('--model', type=str, required=True, help="Model name: 'DeepSNR' or 'D_AEDNet'")
parser.add_argument('--tf', type=str, required=True, help="Transcription factor name (e.g. CDX2)")
parser.add_argument('--motif_len', type=int, default=11, help="Motif length (default: 11)")
args = parser.parse_args()

NeuralNetworkName = args.model
TFsName = args.tf
motif_len = args.motif_len

# === Environment and path setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet
import Utils.Metrics
import Utils.Threshold
from Utils.MotifDiscovery import MotifDiscovery

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"TF: {TFsName}, Model: {NeuralNetworkName}, Motif length: {motif_len}")

data_path = f'../Dataset/ReMap/{TFsName}.bed.txt'
threshold_value = 0.5

# === Load input data ===
FeatureMatrix = Dataset.DataReader.DataReaderPrecit(path=data_path)
FeatureMatrix = np.array(FeatureMatrix)

# === Initialize model ===
if NeuralNetworkName == 'DeepSNR':
    NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
else:
    NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100)

NeuralNetwork.to(device)

TestFeatureMatrix = torch.tensor(FeatureMatrix, dtype=torch.float32).unsqueeze(dim=1)
TestLoader = Dataset.DataLoader.SampleLoaderPredict(FeatureMatrix=TestFeatureMatrix, BatchSize=64)

# === Load trained weights ===
weight_path = f'Weight/{NeuralNetworkName}/{TFsName}_1.pth'
NeuralNetwork.load_state_dict(torch.load(weight_path, map_location=device))
NeuralNetwork.eval()

# === Predict binding positions ===
pred = np.array([])
for step, data in enumerate(tqdm.tqdm(TestLoader, desc="Predicting")):
    X = data.to(device)
    with torch.no_grad():
        logits = NeuralNetwork(X)
        prediction = Utils.Threshold.Threshold(YPredicted=logits, ThresholdValue=threshold_value)
        pred = np.append(pred, prediction)

data_sequence = pd.read_csv(data_path, header=None)
sequence = data_sequence[0].tolist()
label = pred.reshape(-1, 100).astype(np.int64)

# === Extract discovered motifs and save to file ===
output_dir = f'../Algorithm/DiscoveredMotifs/{NeuralNetworkName}/{TFsName}'
os.makedirs(output_dir, exist_ok=True)

outfile_path = os.path.join(output_dir, f'{TFsName}.txt')
seqs = []
with open(outfile_path, 'w') as outfile:
    count = 0
    for idx in tqdm.tqdm(range(len(sequence)), desc="Extracting Motifs"):
        motif = MotifDiscovery(sequence[idx], label[idx], motif_len)
        if 'N' not in motif and len(motif) == motif_len:
            seqs.append(motif)
            outfile.write(motif + '\n')
            count += 1
        if count >= 10000:
            break

# === Generate PWM and WebLogo using Biopython ===
if seqs:
    seqs_for_biopython = [Seq(s) for s in seqs]
    m = motifs.create(seqs_for_biopython)
    m.pseudocounts = 0.01
    m.background = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}

    jaspar_file = os.path.join(output_dir, f"{TFsName}.jaspar")
    with open(jaspar_file, 'w') as f_jaspar:
        f_jaspar.write(m.format("jaspar"))
    print(f"Saved JASPAR PWM to: {jaspar_file}")

    weblogo_file = os.path.join(output_dir, f"{TFsName}_weblogo.png")
    m.weblogo(weblogo_file,
              format='png_print',
              show_fineprint=False,
              show_ends=False,
              color_scheme='color_classic')
    print(f"Saved WebLogo image to: {weblogo_file}")
else:
    print("No valid motifs extracted to create PWM or WebLogo.")
