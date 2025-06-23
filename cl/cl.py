import os
import pandas as pd
import logging
import random
import time
import torch
from datetime import datetime
from datasets import Dataset
from collections import defaultdict
from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
    models,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer
)
import argparse
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers import AutoModel, AutoConfig
from WeightedLayerPooling import WeightedLayerPooling


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
#### /Print debug information to stdout

parser = argparse.ArgumentParser(description="Training script for DNABERT-2 with weighted pooling")
parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--max_seq_length", type=int, default=30, help="Maximum sequence length")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--base_path", type=str, default='/home/yangxu/projectnvme/Model/data/encode690/get_tfbs/159chipseq', help="Base data path")
parser.add_argument("--model_name_or_path", type=str, default="/home/yangxu/projectnvme/Model/DNABERT_2/model/DNABERT-2-117M", help="Pretrained model path")
parser.add_argument("--train_numbers", type=int, default=100, help="Number of training samples")
parser.add_argument("--test_numbers", type=int, default=10, help="Number of test samples")
parser.add_argument("--start_layer", type=int, default=11, help="start_layer for embedding extraction")
parser.add_argument("--model_save_root", type=str, default="output/encode159-ttt-wpol-freeze", help="Root path for saving model checkpoints")

args = parser.parse_args()
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
num_epochs = args.num_epochs
max_seq_length = args.max_seq_length
random_seed = args.random_seed
LR = args.learning_rate
base_path = args.base_path
model_name_or_path = args.model_name_or_path
train_numbers = args.train_numbers
test_numbers = args.test_numbers
start_layer = args.start_layer
model_save_root = args.model_save_root

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

model_save_path = f"{model_save_root}-{train_batch_size}-{num_epochs}-{LR}-{current_time}"

print(f"Base path: {base_path}")
print(f"Model save path: {model_save_path}")
print(f"Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Learning rate: {LR}")


train_datasets = []
test_datasets = []
dev_datasets = []

for folder_name in os.listdir(base_path):
    # Dynamically build file paths
    folder_path = os.path.join(base_path, folder_name)
    train_dataset_path = os.path.join(folder_path, 'train.csv')
    test_dataset_path = os.path.join(folder_path, 'test.csv')
    dev_dataset_path = os.path.join(folder_path, 'dev.csv')
    
    if not os.path.exists(train_dataset_path) or not os.path.exists(test_dataset_path) or not os.path.exists(dev_dataset_path):
        print(f"Missing dataset in folder {folder_name}!")
        continue
    train_datasets.append(train_dataset_path)
    test_datasets.append(test_dataset_path)
    dev_datasets.append(dev_dataset_path)

def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

class Dataprocess:
    def __init__(self, datapaths, dataset_type, train_numbers, test_numbers):
        """
        datapaths: list of file paths
        dataset_type: 'train', 'dev', or 'test'
        train_numbers/test_numbers: 
        target_label: default '1'
        """
        self.datapaths = datapaths
        self.dataset_type = dataset_type
        self.train_numbers = train_numbers
        self.test_numbers = test_numbers
        self.samples = []
        
        self.target_label = '1'

    def seq_load(self, file_handle):
        return [line.strip() for line in file_handle if len(line.strip()) >= 10]

    def _build_label_groups(self, sequences):
        """ dict: {label: [sequence, ...]}"""
        label_groups = defaultdict(list)
        for seq in sequences:
            if ',' not in seq:
                continue  
            sequence, label = seq.rsplit(',', 1)
            label = label.strip()
            sequence = sequence.strip()
            label_groups[label].append(sequence)
        return label_groups

    def forward(self):
        self.samples.clear()
        file_idx = 0
        for path in self.datapaths:
            file_idx += 1
            with open(path, 'rt', encoding='utf8') as f:
                next(f)  # skip header line
                seq_loader = self.seq_load(f)

            label_groups = self._build_label_groups(seq_loader)
            sequences = label_groups.get(self.target_label, [])

            if not sequences:
                continue

            if self.dataset_type == 'train':
                for _ in range(self.train_numbers):
                    if len(sequences) < 2:
                        break  
                    seq1, seq2 = random.sample(sequences, 2)
                    sample_pair = {
                        "sentence_A": seq1,
                        "sentence_B": seq2,
                        "label": -1 
                    }
                    self.samples.append(sample_pair)

            elif self.dataset_type in ['dev', 'test']:
                for _ in range(self.test_numbers):
                    seq = random.choice(sequences)
                    sample = (seq, file_idx)  
                    self.samples.append(sample)

        return self.samples

    def __len__(self):
        return len(self.samples)

def get_eval_samples(samples):
    numbers = len(samples)
    label_groups = {}
    data = []
    
    # label_groups dict
    for sample in samples:
        sequence, label = sample[0], sample[1]
        
        if label not in label_groups:
            label_groups[label] = []
        
        label_groups[label].append(sequence)
    
    keys = list(label_groups.keys())
    print(f"✨The number of keys is {len(keys)}")
    
    for _ in range(numbers):
        random_key_0 = random.choice(keys)
        
        if random.random() > 0.5:
            sentence_A = random.choice(label_groups[random_key_0])
            sentence_B = random.choice(label_groups[random_key_0])
            sample_pair = {"sentence_A": sentence_A, "sentence_B": sentence_B, "label": 1}
            data.append(sample_pair)
        else:
            other_keys = [key for key in label_groups.keys() if key != random_key_0]         
            random_key_1 = random.choice(other_keys)

            sentence_A = random.choice(label_groups[random_key_0])
            sentence_B = random.choice(label_groups[random_key_1])
            sample_pair = {"sentence_A": sentence_A, "sentence_B": sentence_B, "label": 0}
            data.append(sample_pair)
    
    return data

def get_dict(samples):
    samples_dict = {
        "sentence_A": [sample["sentence_A"] for sample in samples],
        "sentence_B": [sample["sentence_B"] for sample in samples],
        "label": [sample["label"] for sample in samples]
        }
    return samples_dict

def shuffle_samples(samples):
    combined_samples = list(zip(samples["sentence_A"], samples["sentence_B"], samples["label"]))
    
    random.shuffle(combined_samples)
    shuffled_samples = {
        "sentence_A": [sample[0] for sample in combined_samples],
        "sentence_B": [sample[1] for sample in combined_samples],
        "label": [sample[2] for sample in combined_samples]
    }
    return shuffled_samples

def evaluate_and_save_results(model, test_dataset, csv_path, stage):
    evaluator = BinaryClassificationEvaluator(
        sentences1=test_dataset["sentence_A"],
        sentences2=test_dataset["sentence_B"],
        labels=test_dataset["label"],
        name=stage
    )
    
    score = evaluator(model, output_path=None)
    
    result_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "score": score
    }
    
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["timestamp", "stage", "score"])
    
    df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
    print(f"[{stage}] Evaluation score: {score}")


# train_samples is a list of InputExample objects, passing the same sentence twice to texts, i.e., texts=[sent, sent]
print('✨train')
train_samples = Dataprocess(train_datasets, 'train', train_numbers, test_numbers).forward()
train_samples = get_dict(train_samples)
train_samples = shuffle_samples(train_samples)

logging.info("reads-test")
print('✨test')

# Test set
test_samples = Dataprocess(test_datasets,'test', train_numbers, test_numbers).forward()
test_samples = get_eval_samples(test_samples)
test_samples = get_dict(test_samples)

print('✨dev')
logging.info("read-dev")
# Development set
dev_samples = Dataprocess(dev_datasets,'dev', train_numbers, test_numbers).forward()
dev_samples = get_eval_samples(dev_samples)
dev_samples = get_dict(dev_samples)


# Convert train_samples to a Dataset
train_dataset = Dataset.from_dict(train_samples)
dev_dataset = Dataset.from_dict(dev_samples)
test_dataset = Dataset.from_dict(test_samples)

logging.info(f"✨ Number of samples in train dataset: {len(train_dataset)}")
logging.info(f"✨ Number of samples in dev dataset: {len(dev_dataset)}")
logging.info(f"✨ Number of samples in test dataset: {len(test_dataset)}")


print("Train Dataset Samples:")
print(train_dataset[:1])
print("Dev Dataset Samples:")
print(dev_dataset[:1])
print("Test Dataset Samples:")
print(test_dataset[:1])


class CustomTransformer(models.Transformer):
    def __init__(self, model_name_or_path, max_seq_length, **kwargs):
        super().__init__(model_name_or_path, max_seq_length, **kwargs)

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    
    def forward(self, features):
        """
        Use the Transformer model to compute hidden embeddings and return all hidden states.
        """
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

        # Use DNABERT-2 to compute hidden layer embeddings
        outputs = self.auto_model(input_ids=input_ids, attention_mask=attention_mask, output_all_encoded_layers=True)
        
        # Retrieve all hidden states from the model
        hidden_states = outputs[0]  # list: num_layers x (batch_size, seq_len, hidden_dim)

        # Store all hidden states in the features dictionary
        features["all_layer_embeddings"] = hidden_states
        
        # Set the last layer's hidden states as token_embeddings (to ensure compatibility with SBERT)
        features["token_embeddings"] = hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)
        
        return features


word_embedding_model = CustomTransformer(model_name_or_path, max_seq_length=max_seq_length)
pooling_model = WeightedLayerPooling(
    word_embedding_dimension = 768,
    num_hidden_layers = 12, 
    layer_start= start_layer, 
    layer_weights=None)

print(f'✨the pooling model is {pooling_model}')

model = SentenceTransformer(modules=[word_embedding_model, pooling_model],trust_remote_code=True)
#print(model)

for param in model[1].parameters():  # model[1] is WeightedLayerPooling 
    param.requires_grad = False

bert_model = model[0].auto_model
#print(bert_model)


#for name, param in model[0].named_parameters():
 #   if param.requires_grad:
  #      print(f"✅Trainable parameter: {name}, Shape: {param.shape}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

train_loss = losses.MultipleNegativesRankingLoss(model)

logging.info("Performance evaluation before training")

evaluator = BinaryClassificationEvaluator(
    sentences1=dev_dataset["sentence_A"],
    sentences2=dev_dataset["sentence_B"],
    labels=dev_dataset["label"],
)

print("Model is now on device:", next(model.parameters()).device)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=model_save_path,
    # Optional training parameters:
    metric_for_best_model="cosine_ap",
    greater_is_better=True,
    num_train_epochs=num_epochs,
    seed = random_seed,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=LR,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=2000,
    run_name=model_save_path,  # Will be used in W&B if `wandb` is installed
)

csv_path = os.path.join(model_save_path, "test_results.csv")
evaluate_and_save_results(model, test_dataset, csv_path, stage="before_training")

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    loss=train_loss,
    evaluator=evaluator
)

trainer.train()
##############################################################################
#
# Load the stored model and evaluate its performance on the STS benchmark dataset
#
##############################################################################

model.save(model_save_path) 

best_model = SentenceTransformer(model_save_path, trust_remote_code=True)
best_model = best_model.to(device)

evaluate_and_save_results(best_model, test_dataset, csv_path, stage="after_training")