# Fake News Detection using Few-Shot Learning and XAI

This project is a research prototype for fake news detection using Natural Language Processing, few-shot learning experiments, and Explainable AI.

## Project Workflow

Input news text → Text preprocessing → Fake/real prediction → LIME XAI explanation → Final output with reason

## Methods Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Few-shot sampling
- LIME XAI

## Dataset

The dataset contains fake and real news articles with the following columns:

- Text
- Label

Labels:

- 0 = Real
- 1 = Fake

## Completed Work

- Dataset loading
- Text preprocessing
- Baseline model training
- Few-shot dataset creation
- Few-shot model training
- LIME explanation
- Terminal-based fake/real prediction system

## Baseline Result

The baseline TF-IDF + Logistic Regression model achieved 94% accuracy on the test dataset.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

## Few-Shot Results

| Shot Setting | Training Samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| 4-shot | 8 | 0.732 | 0.751 | 0.732 | 0.727 |
| 8-shot | 16 | 0.771 | 0.788 | 0.771 | 0.767 |
| 16-shot | 32 | 0.839 | 0.839 | 0.839 | 0.839 |
| 32-shot | 64 | 0.870 | 0.871 | 0.870 | 0.870 |

The 32-shot model achieved the best few-shot performance with 87% accuracy using only 64 labelled training samples.