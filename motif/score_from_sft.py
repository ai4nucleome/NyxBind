import os
import csv
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
import torch.nn.functional as F


class DNADataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer, max_length=30):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sequences[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'offset_mapping': encoded['offset_mapping'].squeeze(0)
        }


def cls_attention_to_base_attention(sequence: str, cls_attention: torch.Tensor, offset_mapping: torch.Tensor, attention_mask: torch.Tensor):
    # Initialize a base_attention vector with zeros
    L = len(sequence)
    base_attention = torch.zeros(L, dtype=torch.float)

    for i, (start, end) in enumerate(offset_mapping.tolist()):
        if attention_mask[i] == 0:
            continue
        if start == end:
            continue
        length = end - start
        if length <= 0:
            continue
        # Evenly distribute the CLS attention score of the current token to the corresponding base positions
        att_value = cls_attention[i].item()
        base_attention[start:end] += att_value / length

    # Perform z-score normalization
    mean = base_attention.mean()
    std = base_attention.std()
    if std > 0:
        base_attention = (base_attention - mean) / std
    else:
        base_attention = base_attention - mean  # Avoid division by zero

    return base_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--selected_layers', type=str, default="6", help="Comma-separated layer indices, e.g. 4,5,6")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_name = os.path.basename(args.data_path.rstrip("/"))
    output_dir = os.path.join(args.output_root, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving attention outputs to: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True,local_files_only=True)
    model.to(device)
    model.eval()

    # Load data
    sequences = []
    labels = []
    data_file = os.path.join(args.data_path, 'motif.csv')
    if not os.path.exists(data_file):
        print(f"Data file does not exist: {data_file}")
        return

    with open(data_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["sequence"])
            labels.append(int(row["label"]))

    dataset = DNADataset(sequences, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Parse selected layers
    selected_layers = list(map(int, args.selected_layers.split(',')))

    all_base_attentions = [None] * len(sequences)
    all_probabilities = [None] * len(sequences)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="✨Processing batches", colour='magenta')):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offset_mapping = batch["offset_mapping"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            attentions = outputs["attentions"]  # list of tensors: [num_layers x (batch_size, heads, seq_len, seq_len)]

            layer_attns = [attentions[layer].mean(dim=1) for layer in selected_layers]  # Average over all heads: [batch, seq_len, seq_len]
            mean_attn = torch.stack(layer_attns).mean(dim=0)  # Sum across different layers: [batch, seq_len, seq_len]

            # Extract CLS token attention scores (index 0)
            cls_token_idx = 0  # CLS token in the sequence
            cls_token_attention = mean_attn[:, cls_token_idx]  # CLS token attention scores

            for i in range(cls_token_attention.size(0)):
                seq_idx = batch_idx * args.batch_size + i
                if seq_idx < len(sequences):
                    seq = sequences[seq_idx]
                    cls_att = cls_token_attention[i]  # Get the CLS token's attention score
                    offsets = offset_mapping[i]
                    mask = attention_mask[i].cpu()

                    # Map CLS attention scores to base attention (every base receives an equal share of the CLS score)
                    base_att = cls_attention_to_base_attention(seq, cls_att, offsets, mask)
                    all_base_attentions[seq_idx] = base_att.numpy()
                    all_probabilities[seq_idx] = probs[i]

    all_attention_filename = os.path.join(output_dir, "atten.npy")
    np.save(all_attention_filename, np.array(all_base_attentions))

    print(f"All attention scores saved to: {all_attention_filename}")


if __name__ == "__main__":
    main()
