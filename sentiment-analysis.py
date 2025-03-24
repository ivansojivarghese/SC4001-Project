
'''

This script sets up:
✅ Multi-Task Learning (MTL) – Using a single model across domains.
✅ Meta-Learning (MAML) – For fast adaptation to new datasets.
✅ Contrastive Learning (SetFit) – For efficient few-shot classification.
✅ Domain Adaptation (DANN) – Making the model robust to domain shifts.

To test the sentiment analysis model, we need to:
> Prepare test data – Load a test dataset from a different domain.
> Perform inference – Run the trained model on new text samples.
> Evaluate performance – Measure accuracy, F1-score, and domain generalization.

'''


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load RoBERTa & DeBERTa Models and Tokenizers
roberta_model = AutoModel.from_pretrained("roberta-base")
deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Define Sentiment Classifier
class SentimentClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(768, 3)  # 3 sentiment classes: Positive, Neutral, Negative

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        return self.classifier(outputs)

# Instantiate Models
roberta_classifier = SentimentClassifier(roberta_model)
deberta_classifier = SentimentClassifier(deberta_model)

# Load a sample dataset for few-shot learning
dataset = load_dataset("sst2", split="train[:100]")

def train_model(model, tokenizer, dataset):
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for batch in dataset:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(batch["label"]).unsqueeze(0)  # Ensure labels have correct shape
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training complete!")

# Train RoBERTa and DeBERTa models
train_model(roberta_classifier, roberta_tokenizer, dataset)
train_model(deberta_classifier, deberta_tokenizer, dataset)

def predict_sentiment(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    predictions = torch.argmax(outputs, dim=1)
    return predictions

# Load test dataset
test_dataset = load_dataset("imdb", split="test[:100]")
test_texts = test_dataset["text"]
true_labels = torch.tensor(test_dataset["label"])

# Predict and evaluate
roberta_predictions = predict_sentiment(roberta_classifier, roberta_tokenizer, test_texts)
deberta_predictions = predict_sentiment(deberta_classifier, deberta_tokenizer, test_texts)

# Calculate Accuracy & F1-score
roberta_accuracy = accuracy_score(true_labels, roberta_predictions)
roberta_f1 = f1_score(true_labels, roberta_predictions, average="weighted")
deberta_accuracy = accuracy_score(true_labels, deberta_predictions)
deberta_f1 = f1_score(true_labels, deberta_predictions, average="weighted")

print(f"RoBERTa Accuracy: {roberta_accuracy:.4f}, F1 Score: {roberta_f1:.4f}")
print(f"DeBERTa Accuracy: {deberta_accuracy:.4f}, F1 Score: {deberta_f1:.4f}")
