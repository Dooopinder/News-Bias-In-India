# News-Bias-In-India
Political Bias Detection In News Articles


This project uses traditional ML and BERT-based models to classify Indian English-language news as **Left-wing**, **Right-wing**, or **Neutral**. It incorporates sentiment analysis and manual labeling to fine-tune models for real-world applicability.

---

## 🚀 Features
- Accepts CSV or article text input
- CLI-based inference interface
- BERT fine-tuned for political alignment
- Stereo/anti-stereo sentiment analysis model
- Supports probability-based predictions with confidence warnings

---

## 🧠 Models Used
- `bert_bias_tone_model` – Trained on [`IndiBias_v1_sample.csv`](https://github.com/sahoonihar/IndiBias) for sentiment tone (stereo/anti-stereo)
- `bert_political_bias_model` – Trained on 1000 manually labeled Indian articles (sourced via Mediastack API + manual annotation)



---

## ⚙️ Requirements
- Python 3.10+
- Libraries: `transformers`, `torch`, `datasets`, `nltk`, `TextBlob`, `joblib`, `pandas`

Install dependencies:
```bash
pip install -r requirements.txt
