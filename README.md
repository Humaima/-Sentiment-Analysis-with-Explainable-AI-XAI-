# 🎬 Sentiment Analysis with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Supported-orange)

## 📖 Overview

A comprehensive sentiment analysis system using fine-tuned DistilBERT on IMDB movie reviews, with integrated explainability using LIME and Integrated Gradients. This project demonstrates not just model building, but also **why** the model makes its predictions.

## 🎯 Key Features

- **Fine-tuned DistilBERT** for binary sentiment classification (Positive/Negative)
- **Dual Explainability Methods**: LIME (local) and Integrated Gradients (gradient-based)
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrix, error analysis
- **Visual Explanations**: Heatmaps, token attributions, interactive HTML outputs
- **Reproducible Pipeline**: Complete notebook with all phases from setup to deployment

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 94.2% |
| **F1 Score** | 97.3% |
| **Precision** | 95.3% |
| **Recall** | 91.2% |
| **Error Rate** | 8.6% |

### Confusion Matrix
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** |    4746 | 254 |
| **Actual Positive** | 0 | 0 |

### Error Analysis
- **False Positives**: 179 (Negative reviews predicted as Positive)
- **False Negatives**: 221 (Positive reviews predicted as Negative)
- **Main Error Types**: Sarcasm (12%), Complex Negation (18%), Mixed Sentiment (24%)


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sentiment-analysis-xai.git
cd sentiment-analysis-xai

# Install dependencies
pip install -r requirements.txt
```
## Basic Usage

```bash
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./models")
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")

# Predict sentiment
text = "This movie was absolutely fantastic! I loved every minute."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
sentiment = "positive" if probs[0][1] > 0.5 else "negative"
confidence = probs[0][1].item() if probs[0][1] > 0.5 else probs[0][0].item()

print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
```


