# Sentiment Analysis Using Transformer-Based Meta-Learning Techniques

## 📘 Overview

This project explores advanced sentiment analysis techniques using transformer-based models in few-shot and multi-domain settings. It was developed for the SC4001 Neural Networks & Deep Learning module at NTU.

## 🧠 Models & Techniques

- DistilBERT: Used as the baseline transformer.
- SetFit: Contrastive learning for few-shot classification.
- MAML: Meta-learning for fast adaptation with minimal data.
- DANN: Domain-Adversarial training for cross-domain generalization.
- Multi-Task Learning: Simultaneous training on IMDb and SST-2 datasets.

## 📊 Datasets

- IMDb: Movie review sentiment (positive/negative).
- SST-2: Stanford Sentiment Treebank (binary sentiment).

## 🧪 Key Experiments

- Full training, few-shot (5-shot), and domain adaptation tests.
- Evaluation using accuracy, F1, ROC-AUC, and confidence-based metrics.
- All models trained and evaluated on Google Colab with GPU acceleration.

## 🏆 Results

- SetFit achieved 100% accuracy in 5-shot settings for both datasets.
- MAML showed strong adaptability but slower training.
- DANN was robust to domain shifts but inconsistent on diverse data.
- Multi-Task Learning was moderately effective; Sequential Fine-Tuning underperformed.

## ✅ Conclusion

SetFit emerged as the best model for low-resource sentiment classification. Combining contrastive learning and meta-learning strategies enables high accuracy, fast training, and strong generalization in real-world settings.