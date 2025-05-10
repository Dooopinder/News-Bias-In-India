# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# --- Load model, tokenizer, label encoder ---
model_path = "./bert_political_bias_model"  # or use "./bert_political_bias_model_v8" if versioned
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
label_encoder = joblib.load("label_encoder.pkl")  # Update if stored elsewhere
model.eval()

# --- Prediction function ---
def classify_political_bias(text):
    if len(text.strip().split()) < 4:
        text = f"The article suggests that {text.strip().lower()} in recent political developments."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = torch.argmax(logits).item()
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    print("\n Prediction Probabilities:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {probs[i]*100:.2f}%")

    if abs(probs[pred_idx] - sorted(probs)[-2]) < 0.15:
        print("\n  Low confidence: Multiple labels are similarly likely.")

    print(f"\n Predicted Political Bias: **{pred_label}**")

# --- CLI Loop ---
def run_cli():
    print("  Political Bias Classifier (BERT-based)")
    print("=========================================")

    while True:
        print("\nChoose input type:")
        print("1. Title only")
        print("2. Full Content")
        print("3. Exit")
        choice = input("Enter 1 / 2 / 3: ").strip()

        if choice == '1':
            user_input = input(" Enter article title: ")
            classify_political_bias(user_input)

        elif choice == '2':
            user_input = input(" Enter full article content: ")
            classify_political_bias(user_input)

        elif choice == '3':
            print(" Exiting. Thank you for using the classifier!")
            break

        else:
            print(" Invalid choice. Please try again.")

# --- Run ---
if __name__ == "__main__":
    run_cli()
