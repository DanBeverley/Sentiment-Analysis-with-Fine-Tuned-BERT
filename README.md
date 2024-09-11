# Sentiment-Analysis-with-Fine-Tuned-BERT

This repository contains a sentiment analysis using a fine-tuned BERT model on the **Google Play Store review dataset**. BERT (Bidirectional Encoder Representations from Transformers) is a powerful transformer-based model for natural language understanding tasks, and this project demonstrates its effectiveness in classifying reviews as positive or negative.

## Overview

In this project, we fine-tune a pre-trained BERT model from Hugging Face's `transformers` library to classify reviews from the Google Play Store as **positive** or **negative**. This can help app developers understand user feedback and improve the overall user experience.

### Key Steps:

1. **Data Preprocessing**: Cleaning and preparing the Google Play Store review dataset.
2. **Model Training**: Fine-tuning a BERT model on the processed review dataset.
3. **Evaluation**: Evaluating the model performance using accuracy, F1 score, precision, and recall.

## Dataset

The dataset contains reviews of apps on the Google Play Store, labeled as either positive or negative. You can download the dataset from [Google Play Store Dataset]([https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews]).

For this project, a balanced sample of positive and negative reviews was used to fine-tune the model.

## Installation

To run this project, you'll need Python 3.x and the required Python libraries listed in `requirements.txt`. 

## Data Preprocessing

Before training the model, the dataset needs to be preprocessed. This involves:
- Tokenizing the reviews using the BERT tokenizer.
- Padding and truncating reviews to ensure uniform input size.
- Converting reviews into BERT's input format (attention masks, segment IDs).


## Model Fine-Tuning

We fine-tune a pre-trained BERT model (`bert-base-uncased`) from the Hugging Face `transformers` library on the Google Play Store review dataset.

The training script includes:
- Loading the pre-trained BERT model.
- Adding a classification head for binary sentiment classification.
- Using a cross-entropy loss function to fine-tune the model.

## Usage

If you'd like to use the fine-tuned model to predict sentiment on new reviews, you can run the following:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/fine_tuned_bert/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and predict
review = "This app is amazing!"
inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()

sentiment = "Positive" if prediction == 1 else "Negative"
print(f"Review sentiment: {sentiment}")
```

## Results

After fine-tuning, the model achieved the following performance on the test set:
- **Accuracy**: 89.5%
- **F1 Score**: 90.2%
- **Precision**: 88.7%
- **Recall**: 91.6%
