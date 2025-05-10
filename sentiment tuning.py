# -*- coding: utf-8 -*-
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import os

# ✅ Disable WANDB logging
os.environ["WANDB_DISABLED"] = "true"

# Step 1: Load and prepare dataset
df_bias = pd.read_csv("IndiBias_v1_sample.csv", encoding="utf-8")
df_bias = df_bias.rename(columns={"modified_eng_sent_more": "text"})
df_bias["label"] = df_bias["stereo_antistereo"].apply(lambda x: 1 if x == "stereo" else 0)
df_bias = df_bias[["text", "label"]]

# Step 2: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df_bias)

# Step 3: Tokenize
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Step 4: Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 5: Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Step 6: Training arguments (without evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./bert_bias_tone_model",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    report_to="none"
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 8: Train
trainer.train()

# Step 9: Evaluate manually
metrics = trainer.evaluate()
print("\n Final Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Step 10: Save model
model.save_pretrained("./bert_bias_tone_model")
tokenizer.save_pretrained("./bert_bias_tone_model")






'''

import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your saved tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("./bert_bias_tone_model")
model = BertForSequenceClassification.from_pretrained("./bert_bias_tone_model")

# Load and prepare your evaluation dataset
df = pd.read_csv("IndiBias_v1_sample.csv")
df = df.rename(columns={"modified_eng_sent_more": "text"})
df["label"] = df["stereo_antistereo"].apply(lambda x: 1 if x == "stereo" else 0)
df = df[["text", "label"]]
dataset = Dataset.from_pandas(df)

# Tokenize
def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Setup Trainer just for evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate
metrics = trainer.evaluate(eval_dataset=dataset["test"])
print("\n📊 Sentiment Model Evaluation:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

'''













































