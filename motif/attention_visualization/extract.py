"""
Extract raw attention weights from a transformer model.

This script extracts attention weights from all layers of a pre-trained model
(e.g., DNABERT-2, NyxBind) for a given DNA sequence and saves them as a numpy array.
"""

import torch
import numpy as np
import os
import argparse
from transformers import AutoTokenizer, AutoModel


def extract_raw_attention(sequence, model_path, output_dir="./attention_output", max_length=30, save_name="raw_attention.npy"):
    """
    Extract attention weights from all layers of a transformer model.
    
    Args:
        sequence (str): Input DNA sequence
        model_path (str): Path to the pre-trained model
        output_dir (str): Directory to save the attention output
        max_length (int): Maximum token length for the sequence
        save_name (str): Filename for the saved attention array (.npy)
    
    Returns:
        tuple: (attention_array, tokenizer, encoded_input)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, output_attentions=True, local_files_only=True)
    model.to(device)
    model.eval()

    # Tokenize input sequence
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True, return_dict=True)

    # Handle both dict and ModelOutput formats
    if isinstance(outputs, dict):
        attentions = outputs.get("attentions", None)
    else:
        attentions = getattr(outputs, "attentions", None)

    if attentions is None:
        raise ValueError("The model did not return attention weights. Make sure output_attentions=True works for this model.")

    num_layers = len(attentions)
    print(f"✓ Extracted {num_layers} layers of attention.")

    # Convert to numpy and save
    all_layer_attn = [att.cpu().numpy() for att in attentions]
    all_layer_attn = np.array(all_layer_attn)  # shape: [num_layers, 1, heads, seq_len, seq_len]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, save_name)
    np.save(out_path, all_layer_attn)

    print(f"✓ Attention saved to: {out_path}")
    print(f"  Shape: {all_layer_attn.shape} (layers, batch=1, heads, seq_len, seq_len)")

    return all_layer_attn, tokenizer, encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all-layer raw attention from one sequence")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--sequence", type=str, required=True, help="DNA or text sequence")
    parser.add_argument("--output_dir", type=str, default="./attention_output", help="Output directory")
    parser.add_argument("--max_length", type=int, default=30, help="Max token length")
    parser.add_argument("--save_name", type=str, default="raw_attention.npy", help="Filename to save attention (including .npy)")

    args = parser.parse_args()

    extract_raw_attention(
        sequence=args.sequence,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        save_name=args.save_name
    )
