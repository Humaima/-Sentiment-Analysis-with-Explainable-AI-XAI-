# 🎬 Sentiment Analysis with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00)

A hybrid deep learning model for sentiment analysis that combines BERT's contextual understanding with LSTM's sequential processing capabilities, enhanced with LIME for model interpretability and explainability.

## 📋 Overview

This project implements an explainable sentiment analysis system that:
- Uses a BERT-LSTM hybrid architecture for improved text classification
- Incorporates LIME (Local Interpretable Model-agnostic Explanations) for model interpretability
- Provides visual explanations of model predictions
- Supports deployment and sharing via Hugging Face Hub

## 🏗️ Architecture

### Model Architecture
1. **BERT Encoder**: Pretrained BERT-base model for contextual embeddings
2. **LSTM Layer**: Captures sequential dependencies in the encoded representations
3. **Classification Head**: Fully connected layers for sentiment prediction
4. **LIME Explainer**: Generates local explanations for model predictions

### Key Features
- ✅ Hybrid BERT-LSTM architecture for enhanced performance
- ✅ Model interpretability with LIME explanations
- ✅ Visual heatmaps for feature importance
- ✅ Support for multiple sentiment classes
- ✅ Easy deployment to Hugging Face Hub
- ✅ Comprehensive training pipeline with checkpointing

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/yourusername/explainable-sentiment-analysis.git
cd explainable-sentiment-analysis

# Install requirements
pip install -r requirements.txt
```

### Required Packages

```bash
torch>=2.0.0
transformers>=4.30.0
lime>=0.2.0.1
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
huggingface-hub>=0.15.0
datasets>=2.12.0
accelerate>=0.20.0
```

## 📁 Project Structure

```bash
explainable-sentiment-analysis/
│
├── notebooks/
│   └── Explainable_Sentiment_Analysis_using_BERT_LSTM_+_LIME_.ipynb
│
├── data/
│   ├── train_data.csv         # Training dataset
│   ├── test_data.csv          # Testing dataset
│   └── processed/             # Processed data files
│
├── models/
│   ├── best_model.pt          # Best trained model weights
│   ├── checkpoint_epoch_1.pt  # Training checkpoints
│   ├── checkpoint_epoch_2.pt
│   ├── checkpoint_epoch_3.pt
│   └── model.safetensors      # Hugging Face compatible format
│
├── outputs/
│   ├── training_history.png   # Training metrics visualization
│   └── explanations/          # Generated LIME explanations
│
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
└── README.md                 # This file
```
## 🎯 Usage

### 1. Data Preparation

```python
from src.data_loader import SentimentDataset

# Load and preprocess data
dataset = SentimentDataset(
    data_path="data/train_data.csv",
    tokenizer_name="bert-base-uncased",
    max_length=128
)
```
### 2. Model Training
```bash
from src.train import SentimentTrainer

trainer = SentimentTrainer(
    model_name="bert-base-uncased",
    num_classes=3,
    lstm_hidden_size=256,
    learning_rate=2e-5,
    batch_size=32
)

# Train the model
trainer.train(
    train_dataset=dataset,
    epochs=5,
    save_dir="models/"
)
```
### 3. Generate Explanations
```bash
from src.explain import LIMEExplainer

explainer = LIMEExplainer(model_path="models/best_model.pt")
text = "The movie was absolutely fantastic with great acting!"

# Generate explanation
explanation = explainer.explain(
    text=text,
    num_features=10,
    num_samples=5000
)

# Visualize explanation
explainer.visualize(explanation, save_path="outputs/explanations/")
```

