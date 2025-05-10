
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Disable WANDB logging
os.environ["WANDB_DISABLED"] = "true"

# Load your labeled dataset
df = pd.read_csv("manually_labeled_1000.csv")

# Combine Title + Description if available
if "Title" in df.columns and "Description" in df.columns:
    df["text"] = df["Title"].fillna('') + ". " + df["Description"].fillna('')
else:
    df = df.rename(columns={"Content": "text"})


# Encode using LabelEncoder and save it
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Manual Bias"])
joblib.dump(label_encoder, "label_encoder.pkl")


# Stratified split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)



# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df["label"]), y=df["label"])
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("./bert_bias_tone_model")  # Use sentiment model's tokenizer

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Model initialized from Step 1 sentiment model
model = BertForSequenceClassification.from_pretrained(
    "./bert_bias_tone_model", 
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Apply class weights to loss function
model.classifier = torch.nn.Linear(model.config.hidden_size, 3)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='macro'),
        "recall": recall_score(labels, preds, average='macro'),
        "f1": f1_score(labels, preds, average='macro')
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_political_bias_model",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    report_to="none",
    load_best_model_at_end=False,  #  Only once
    fp16=True
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print("\n  Final Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save final model
model.save_pretrained("./bert_political_bias_model")
tokenizer.save_pretrained("./bert_political_bias_model")








'''
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the labeled dataset used for training
df = pd.read_csv("manually_labeled_1000.csv")
df = df.rename(columns={"Content": "text", "Manual Bias": "label"})

# Encode string labels to numeric (must match training)
label_mapping = {"Left-wing": 0, "Right-wing": 1, "Neutral": 2}
df["label"] = df["label"].map(label_mapping)

# Step 2: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.2)

# Step 3: Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("./bert_political_bias_model")
model = BertForSequenceClassification.from_pretrained("./bert_political_bias_model")

# Step 4: Tokenize
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

dataset = dataset.map(tokenize_fn, batched=True)

# Step 5: Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='macro'),
        "recall": recall_score(labels, preds, average='macro'),
        "f1": f1_score(labels, preds, average='macro')
    }

# Step 6: Evaluate
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

metrics = trainer.evaluate(dataset["test"])

# Step 7: Print results
print("\n📊 Political Bias Model Evaluation:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

'''












































