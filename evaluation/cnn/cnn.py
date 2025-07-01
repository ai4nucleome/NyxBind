import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import torch
import transformers
from transformers import AutoModel, Trainer, EarlyStoppingCallback
import torch.nn as nn
import sklearn
import numpy as np
from torch.utils.data import Dataset
import os
from DanQ import DanQ  
from DeepBind import DeepBind

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    nummotif: int = field(default=16, metadata={"help": "Number of motifs."})
    motiflen: int = field(default=24, metadata={"help": "Motif length."})
    poolType: str = field(default="max", metadata={"help": "Pooling type: max or avg."})
    neuType: str = field(default="hidden", metadata={"help": "Neural layer type: hidden or other."})
    mode: str = field(default="training", metadata={"help": "Mode: training or testing."})
    dropprob: float = field(default=0.3, metadata={"help": "Dropout probability."})
    sigmaConv: float = field(default=0.1, metadata={"help": "Sigma for conv layer."})
    sigmaNeu: float = field(default=0.1, metadata={"help": "Sigma for dense layer."})
    reverse_complement_mode: bool = field(default=False, metadata={"help": "Use reverse complement or not."})



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    reverse_complement_enabled: bool = field(default=True, metadata={"help": "Use reverse complement or not."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: Optional[str] = field(default="linear")
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps"),  # or "epoch"
    warmup_steps: int = field(default=30)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    greater_is_better: bool = field(default=False)
    metric_for_best_model: str = field(default="eval_loss")  

def reverse_complement(sequence: str) -> str:
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement[base] for base in reversed(sequence))

def one_hot_encode_sequence(sequence: str, alphabet="ACGT") -> torch.Tensor:
    encoding = {char: idx for idx, char in enumerate(alphabet)}
    one_hot = torch.zeros(len(sequence), len(alphabet), dtype=torch.float32)
    for i, char in enumerate(sequence):
        if char in encoding:
            one_hot[i, encoding[char]] = 1.0
    return one_hot

class SupervisedDataset(Dataset):
    """One-hot encoded sequence dataset for supervised fine-tuning (both forward and reverse-complement strands are included as separate samples)."""

    def __init__(self, data_path: str, reverse_complement_enabled=False):
        super(SupervisedDataset, self).__init__()

        self.reverse_complement_enabled = reverse_complement_enabled

        # Load data from CSV (skipping the header)
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        # Check data format
        if len(data[0]) == 2:
            logging.warning("Running single-sequence classification task (using one-hot encoding)...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        else:
            raise ValueError("Unsupported data format. Expected format: [sequence, label]")

        # Initialize sequence and label lists
        self.sequences = []
        self.labels = []

        for seq, label in zip(texts, labels):
            one_hot = one_hot_encode_sequence(seq)
            self.sequences.append(one_hot)
            self.labels.append(label)

            # Optionally add reverse-complement sequence
            if self.reverse_complement_enabled:
                rev_comp_seq = reverse_complement(seq)
                rev_comp_one_hot = one_hot_encode_sequence(rev_comp_seq)
                self.sequences.append(rev_comp_one_hot)
                self.labels.append(label)

        self.num_labels = len(set(self.labels))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = self.sequences[i].transpose(0, 1)  # shape: (4, L)
        label = torch.tensor(self.labels[i], dtype=torch.long).unsqueeze(0)

        return dict(input_ids=input_ids, labels=label)


@dataclass
class DataCollatorForSupervisedDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print("DataCollatorForSupervisedDataset called with:")
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.tensor(labels).long()
        #print("labels shape:", labels.shape)
        return dict(input_ids=input_ids, labels=labels)

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray = None):
    #print(predictions.shape)
    #print(probabilities)
   # print(probabilities.shape)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    roc_auc = sklearn.metrics.roc_auc_score(valid_labels, probabilities[valid_mask])
    pr_auc = sklearn.metrics.average_precision_score(valid_labels, probabilities[valid_mask])
    accuracy = sklearn.metrics.accuracy_score(valid_labels, valid_predictions)
    f1 = sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)
    matthews_correlation = sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions)
    precision = sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0)
    recall = sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "matthews_correlation": matthews_correlation,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.squeeze() 


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.tensor(logits).numpy() 
    preds = (probs > 0.5).astype(int)
    return calculate_metric_with_sklearn(preds, labels, probs)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    modelname = model_args.model_name_or_path

    if modelname == 'DanQ':
        model = DanQ()
    elif modelname == 'DeepBind':
        model = DeepBind()
    else:
        raise ValueError(f"[ERROR] Unsupported model name: '{modelname}'. "
                         f"Available options are: ['DanQ', 'DeepBind']")
    
    # Tokenizer is not needed since we're using one-hot encoding
    tokenizer = None

    # Load training dataset
    train_dataset = SupervisedDataset(
        data_path=os.path.join(data_args.data_path, "train.csv"),
        reverse_complement_enabled=data_args.reverse_complement_enabled  
    )
    print(f"train_dataset: {len(train_dataset)}")
    #for i in range(min(1, len(train_dataset))):
    #    print(f"Sample {i}: {train_dataset[i]}")

    # Load validation dataset
    val_dataset = SupervisedDataset(
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        reverse_complement_enabled=data_args.reverse_complement_enabled
    )
    print(f"val_dataset: {len(val_dataset)}")

    # Load test dataset
    test_dataset = SupervisedDataset(
        data_path=os.path.join(data_args.data_path, "test.csv"),
        reverse_complement_enabled=data_args.reverse_complement_enabled
    )
    print(f"test_dataset: {len(test_dataset)}")

    data_collator = DataCollatorForSupervisedDataset()

    # Check data_collator output shape with a small batch
    #sample_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    #processed_batch = data_collator(sample_batch)
    #print("Processed batch 'input_ids' shape:", processed_batch['input_ids'].shape)
    #print("Processed batch 'labels' shape:", processed_batch['labels'].shape)

    # Check if model has trainable parameters
    print("Checking trainable parameters...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad is not None)

    # Print model architecture
    print("Model architecture:\n")
    print(model)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    trainer.train()

    # Save evaluation results on test set if enabled
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)
        logging.warning(f"Evaluation results saved to {os.path.join(results_path, 'eval_results.json')}")


if __name__ == "__main__":
    train()
