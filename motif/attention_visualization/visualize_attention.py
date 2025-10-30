#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize attention weights as line diagrams.

This script creates horizontal line-based visualizations of attention patterns
from transformer models, showing how tokens attend to each other across layers.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from transformers import AutoTokenizer

def plot_attention_lines_horizontal(ax, attention_matrix, tokens, max_tokens=30, threshold=0.3,
                                    show_left_tokens=False, show_right_tokens=False):
    """
    Plot attention weights as horizontal lines between tokens.
    
    Args:
        ax: Matplotlib axis object
        attention_matrix: Attention weight matrix (seq_len x seq_len)
        tokens: List of token strings
        max_tokens: Maximum number of tokens to display
        threshold: Minimum attention weight to display as a line
        show_left_tokens: Whether to show token labels on the left
        show_right_tokens: Whether to show token labels on the right
    
    Note: Tokens are displayed in reverse order (CLS at bottom, SEP at top)
    """
    num_tokens = min(len(tokens), max_tokens)
    attention = attention_matrix[:num_tokens, :num_tokens]

    ax.set_axis_off()

    # Reorder tokens: [CLS] + middle tokens + [SEP], then reverse
    middle_tokens = [t for t in tokens if t not in ("[CLS]", "[SEP]")]
    tokens_ordered = ["[CLS]"] + middle_tokens + ["[SEP]"]
    tokens_ordered = tokens_ordered[::-1]  # Reverse order: CLS at bottom

    y = np.arange(len(tokens_ordered))
    margin = 0.05  # Left and right margin
    x_left = np.full(len(tokens_ordered), margin)      # Left line start
    x_right = np.full(len(tokens_ordered), 1 - margin) # Right line end

    # Draw token text labels
    for i, t in enumerate(tokens_ordered):
        color = "yellow" if t in ("[CLS]", "[SEP]") else "white"
        if show_left_tokens:
            ax.text(-0.05, y[i], t, fontsize=6, ha='right', va='center', color=color)
        if show_right_tokens:
            ax.text(1.05, y[i], t, fontsize=6, ha='left', va='center', color=color)

    # Draw attention lines
    lines, colors, widths = [], [], []
    for i, t_i in enumerate(tokens_ordered):
        for j, t_j in enumerate(tokens_ordered):
            orig_i = tokens.index(t_i)
            orig_j = tokens.index(t_j)
            w = attention[orig_i, orig_j]
            if w > threshold:
                lines.append([(x_left[i], y[i]), (x_right[j], y[j])])

                scale = 5
                alpha = min(0.8, w * scale)
                lw = min(1, w * scale)
                colors.append((0, 0.8, 1, alpha))  # Cyan with varying alpha
                widths.append(lw)

    if lines:
        lc = LineCollection(lines, colors=colors, linewidths=widths)
        ax.add_collection(lc)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(tokens_ordered))
    ax.set_facecolor("black")


def main():
    """
    Main function to visualize attention from a saved numpy array.
    """
    parser = argparse.ArgumentParser(description="Visualize attention patterns from a saved numpy array")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to attention numpy file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HuggingFace model (for tokenizer)")
    parser.add_argument("--sequence", type=str, required=True, help="Input DNA sequence")
    parser.add_argument("--save_path", type=str, default="attention.svg", help="Output file path")
    parser.add_argument("--threshold", type=float, default=0.02, help="Threshold for displaying attention lines")
    args = parser.parse_args()

    # Load attention numpy array
    attentions = np.load(args.npy_path, allow_pickle=True)
    print(f"✓ Loaded attention shape: {attentions.shape}")

    # Process attention shape
    if attentions.ndim == 5:
        layer_mean_attn = attentions.mean(axis=(1, 2))  # Average over batch and heads
    elif attentions.ndim == 4:
        layer_mean_attn = attentions.mean(axis=1)       # Average over heads
    else:
        raise ValueError(f"Unexpected attention shape: {attentions.shape}")

    num_layers, seq_len, _ = layer_mean_attn.shape
    print(f"✓ Processed {num_layers} layers, each with {seq_len} tokens")

    # Get tokens using tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokens = tokenizer.tokenize(args.sequence)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    # Create horizontal subplots (one per layer)
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, len(tokens)/4))
    if num_layers == 1:
        axes = [axes]

    # Subplot layout parameters
    total_width = 0.6
    left_margin = 0   # Overall left margin
    right_margin = 0  # Overall right margin
    gap = 0           # Gap between subplots
    axes_width = (total_width - left_margin - right_margin - gap*(num_layers-1)) / num_layers

    # Plot attention for each layer
    for i, ax in enumerate(axes):
        show_left = (i == 0)             # Show tokens on left for first plot
        show_right = (i == num_layers-1) # Show tokens on right for last plot
        plot_attention_lines_horizontal(
            ax, layer_mean_attn[i], tokens,
            max_tokens=len(tokens),
            threshold=args.threshold,
            show_left_tokens=show_left,
            show_right_tokens=show_right
        )
        # Optional: ax.set_title(f"Layer {i}", color='white', fontsize=10)

        # Set subplot position with uniform margins
        left = left_margin + i * (axes_width + gap)
        ax.set_position([left, 0.05, axes_width, 0.9])  # [left, bottom, width, height]

    # Set background to black
    fig.patch.set_facecolor("black")
    plt.savefig(args.save_path, format='svg', bbox_inches='tight', facecolor="black")
    plt.close()
    print(f"✓ Attention visualization saved to: {args.save_path}")


if __name__ == "__main__":
    main()
