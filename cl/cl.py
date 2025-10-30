import os
import pandas as pd
import logging
import torch
from datetime import datetime
from datasets import Dataset
from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
    models,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers import AutoModel, AutoConfig
from WeightedLayerPooling import WeightedLayerPooling
import argparse
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

#### Logging setup
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

#### Argparse
parser = argparse.ArgumentParser(description="Training script for DNABERT-2 with weighted pooling")
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_length", type=int, default=30)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--start_layer", type=int, default=11)
parser.add_argument("--model_save_root", type=str, default="output/model")
parser.add_argument("--loss_name", type=str, default="MNRL",
                    choices=["MNRL", "SMNRL", "GEL", "CL"],
                    help="Choose loss function: MNRL | SMNRL | GEL | CL")

args = parser.parse_args()

# ========================
# Configuration
# ========================
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
num_epochs = args.num_epochs
max_seq_length = args.max_seq_length
random_seed = args.random_seed
LR = args.learning_rate
base_path = args.base_path
model_name_or_path = args.model_name_or_path
start_layer = args.start_layer
model_save_root = args.model_save_root
loss_name = args.loss_name   

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = f"{model_save_root}-{loss_name}-{train_batch_size}-{num_epochs}-{LR}-{current_time}"
os.makedirs(model_save_path, exist_ok=True)

print(f"Base path: {base_path}")
print(f"Model save path: {model_save_path}")
print(f"Using loss: {loss_name}")

# ========================
# Dataset loader
# ========================
class TFCSVLoader:
    """Load a single CSV file with columns: sentence_A, sentence_B, label, TF"""
    def __new__(cls, csv_paths):
        self = super().__new__(cls)
        self.__init__(csv_paths)
        dataset_dict = {
            "sentence_A": self.data["sentence_A"].tolist(),
            "sentence_B": self.data["sentence_B"].tolist(),
            "label": self.data["label"].tolist()
        }
        dataset = Dataset.from_dict(dataset_dict)
        tf_dict = self.data["TF"].to_dict()
        return dataset, tf_dict

    def __init__(self, csv_paths):
        if len(csv_paths) != 1:
            raise ValueError("TFCSVLoader only supports loading one CSV file at a time.")
        path = csv_paths[0]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 4:
            raise ValueError(f"{path} must contain at least 4 columns.")
        df.columns = ["sentence_A", "sentence_B", "label", "TF"]
        self.data = df

# ========================
# Load datasets
# ========================
train_csvs = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith("train.csv")]
dev_csvs = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith("dev.csv")]
test_csvs = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith("test.csv")]

train_dataset, train_TF = TFCSVLoader(train_csvs)
dev_dataset, dev_TF = TFCSVLoader(dev_csvs)
test_dataset, test_TF = TFCSVLoader(test_csvs)
print(test_dataset)
print(f"Train size: {len(train_dataset)}, Dev size: {len(dev_dataset)}, Test size: {len(test_dataset)}")

# ========================
# Model definition
# ========================
class CustomTransformer(models.Transformer):
    """Custom transformer model to extract all hidden layers"""
    def __init__(self, model_name_or_path, max_seq_length, **kwargs):
        super().__init__(model_name_or_path, max_seq_length, **kwargs)
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    
    def forward(self, features):
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        outputs = self.auto_model(input_ids=input_ids, attention_mask=attention_mask, output_all_encoded_layers=True)
        hidden_states = outputs[0]
        features["all_layer_embeddings"] = hidden_states
        features["token_embeddings"] = hidden_states[-1]
        return features

word_embedding_model = CustomTransformer(model_name_or_path, max_seq_length=max_seq_length)
pooling_model = WeightedLayerPooling(
    word_embedding_dimension=768,
    num_hidden_layers=12,
    layer_start=start_layer
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], trust_remote_code=True)

# Freeze pooling layer
for param in model[1].parameters():
    param.requires_grad = False
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Select loss function
# ========================
if loss_name == "MNRL":
    train_loss = losses.MultipleNegativesRankingLoss(model)
elif loss_name == "SMNRL":
    train_loss = losses.MultipleNegativesSymmetricRankingLoss(model)
elif loss_name == "GEL":
    train_loss = losses.GISTEmbedLoss(model)
elif loss_name == "CL":
    train_loss = losses.CosineSimilarityLoss(model)
else:
    raise ValueError(f"Unknown loss name: {loss_name}")

print(f"✅ Using loss: {train_loss.__class__.__name__}")

# ========================
# Validation evaluator
# ========================
evaluator = BinaryClassificationEvaluator(
    sentences1=dev_dataset["sentence_A"],
    sentences2=dev_dataset["sentence_B"],
    labels=dev_dataset["label"],
)

# ========================
# Training arguments
# ========================
args_sbert = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    metric_for_best_model="cosine_ap",
    greater_is_better=True,
    num_train_epochs=num_epochs,
    seed=random_seed,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=LR,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=2000,
    run_name=model_save_path
)

# ========================
# Custom Trainer
# ========================
class MyTrainer(SentenceTransformerTrainer):
    """Custom Trainer using SequentialSampler for ordered batch sampling"""
    def train_dataloader(self):
        batch_sampler = BatchSampler(
            SequentialSampler(self.train_dataset),
            batch_size=self.args.per_device_train_batch_size,
            drop_last=False
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=getattr(self.train_dataset, "collate_fn", None),
            num_workers=0
        )

# ========================
# Test evaluator
# ========================
test_evaluator = BinaryClassificationEvaluator(
    sentences1=test_dataset["sentence_A"],
    sentences2=test_dataset["sentence_B"],
    labels=test_dataset["label"],
    name="test",
    batch_size=512,
)

# ========================
# Evaluate before training
# ========================
print("🔍 Evaluating on test set before training...")
test_score_before = test_evaluator(model, output_path=None)
print(f"✅ Test score before training: {test_score_before}")

# ========================
# Initialize Trainer
# ========================
trainer = MyTrainer(
    model=model,
    args=args_sbert,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    loss=train_loss,
    evaluator=evaluator
)

# ========================
# Training
# ========================
trainer.train()
model.save(model_save_path)
print(f"✅ Model saved at {model_save_path}")

# ========================
# Evaluate after training
# ========================
print("🔍 Evaluating on test set after training...")
test_score_after = test_evaluator(model, output_path=None)
print(f"✅ Test score after training: {test_score_after}")

# ========================
# Save results
# ========================
eval_output_dir = "./evaloutput"
os.makedirs(eval_output_dir, exist_ok=True)
csv_path = os.path.join(eval_output_dir, "test_results.csv")

result_row = {
    "timestamp": current_time,
    "model_path": model_save_path,
    "loss_name": loss_name,
    "test_score_before": test_score_before,
    "test_score_after": test_score_after
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=result_row.keys())

df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"✅ Test results (before & after) saved to: {csv_path}")
