# English-Sinhala Fake News Detection using Few-Shot Learning and XAI

This repository contains a research prototype for detecting fake or misleading
news in English and Sinhala. It combines classical NLP baselines, few-shot
learning, multilingual transformer fine-tuning, and LIME-based Explainable AI.

## Research Objectives

The objectives of this research are to:

1. Compare classical and transformer-based fake-news classification methods.
2. Study fake-news detection under limited labelled-data settings.
3. Support English and Sinhala using one multilingual model.
4. Display the confidence of each prediction.
5. Explain individual predictions using LIME word-importance weights.

## System Workflow

```text
English or Sinhala news text
            |
            v
Text cleaning and language detection
            |
            v
XLM-RoBERTa bilingual classifier
            |
            v
Fake/misleading and real/credible probabilities
            |
            v
Confidence threshold and final prediction
            |
            v
LIME explanation with supporting and opposing words
```

## Methods and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Few-shot sampling
- SetFit
- DistilRoBERTa
- XLM-RoBERTa-base
- PyTorch
- Hugging Face Transformers
- LIME Explainable AI

## Datasets

### English Dataset

The English dataset contains news articles with fake and real labels.

It was used for:

- TF-IDF and Logistic Regression baseline training
- 4, 8, 16, 32 and 64-shot experiments
- SetFit and DistilRoBERTa experiments
- English-Sinhala bilingual training and evaluation

### Sinhala Dataset

The Sinhala data is based on the
[LIRNEasia Misinformation Corpus Sinhala](https://github.com/LIRNEasia/MisinformationCorpusSinhala).

The original Sinhala dataset contained the following labels:

- CREDIBLE
- FALSE
- PARTIAL
- UNCERTAIN

The labels were converted for binary classification as follows:

| Original label | Binary label | Treatment |
|---|---|---|
| CREDIBLE | real | Retained |
| FALSE | fake | Retained |
| PARTIAL | fake/misleading | Retained |
| UNCERTAIN | — | Excluded |

After cleaning and duplicate removal, 1,110 usable Sinhala records were retained:

- 1,000 real records
- 110 fake/misleading records

The cleaned Sinhala dataset was divided into training, validation and test
datasets.

## Sinhala Dataset Split

| Dataset | Real | Fake/misleading | Total |
|---|---:|---:|---:|
| Training | 640 | 70 | 710 |
| Validation | 160 | 18 | 178 |
| Testing | 200 | 22 | 222 |
| **Total** | **1,000** | **110** | **1,110** |

## Bilingual 64-Shot Dataset

The balanced bilingual training dataset contains 256 samples:

| Language | Fake/misleading | Real/credible | Total |
|---|---:|---:|---:|
| English | 64 | 64 | 128 |
| Sinhala | 64 | 64 | 128 |
| **Total** | **128** | **128** | **256** |

In this research, 64-shot means 64 labelled examples per class. It does not
refer to 64-bit computing.

The Sinhala fake/misleading examples were selected from:

- 16 FALSE articles
- 48 PARTIAL articles

## Completed Work

- English dataset loading
- Text cleaning and preprocessing
- TF-IDF and Logistic Regression baseline training
- English few-shot dataset creation
- English 4, 8, 16, 32 and 64-shot experiments
- SetFit and DistilRoBERTa experiments
- Bias-aware English 64-shot model training
- Sinhala Excel dataset processing
- Sinhala label conversion
- Duplicate removal
- Sinhala train, validation and test splitting
- Balanced Sinhala 64-shot dataset creation
- English-Sinhala bilingual dataset creation
- XLM-RoBERTa CPU training
- Per-language and overall model evaluation
- English-Sinhala interactive prediction
- Confidence-based uncertain output
- English and Sinhala LIME XAI explanations

## Results

### TF-IDF and Logistic Regression Baseline

The TF-IDF and Logistic Regression baseline achieved approximately 94% test
accuracy on the English dataset.

### English Few-Shot Results

| Shot setting | Training samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| 4-shot | 8 | 0.732 | 0.751 | 0.732 | 0.727 |
| 8-shot | 16 | 0.771 | 0.788 | 0.771 | 0.767 |
| 16-shot | 32 | 0.839 | 0.839 | 0.839 | 0.839 |
| 32-shot | 64 | 0.870 | 0.871 | 0.870 | 0.870 |

The 32-shot model achieved 87% accuracy using only 64 labelled training
samples.

### English Bias-Aware 64-Shot SetFit Result

The bias-aware English SetFit model using DistilRoBERTa achieved approximately
92.95% test accuracy.

### Initial Bilingual XLM-R Experiment

The first XLM-R training experiment collapsed into predicting only one class.

This experiment produced approximately:

- Overall accuracy: 40.41%
- Overall macro F1: 28.78%

This result was retained as a failed experiment and used to improve the second
training configuration.

### Improved Bilingual XLM-R v2 Results

The second training configuration unfroze more encoder layers, used separate
learning rates and trained for additional epochs.

| Metric | Result |
|---|---:|
| Overall accuracy | 0.8380 |
| Overall macro F1 | 0.8293 |
| English accuracy | 0.8591 |
| English macro F1 | 0.8565 |
| Sinhala accuracy | 0.7477 |
| Sinhala macro F1 | 0.5889 |
| Mean-language macro F1 | 0.7227 |

### Bilingual Test Confusion Matrices

Overall confusion matrix:

```text
[[359, 115],
 [ 75, 624]]
```

English confusion matrix:

```text
[[345, 107],
 [ 27, 472]]
```

Sinhala confusion matrix:

```text
[[14, 8],
 [48, 152]]
```

The confusion-matrix class order is:

```text
[fake, real]
```

## Explainable AI

LIME is used to explain individual English and Sinhala predictions.

For each prediction, the system displays:

- Words or phrases supporting the predicted class
- Words or phrases opposing the predicted class
- A LIME importance weight for each feature

Positive weights support the predicted class.

Negative weights oppose the predicted class.

Example:

```text
Class being explained: REAL/CREDIBLE

Words supporting REAL/CREDIBLE
President       +0.0939
said            +0.0523
stakeholders    +0.0423

Words opposing REAL/CREDIBLE
Crime           -0.0236
had             -0.0195
```

LIME explains the language patterns that influenced the model. It does not
provide external factual evidence proving that a news claim is true or false.

## Project Structure

```text
Fake-News-Few-Shot-XAI/
│
├── data/
│   ├── raw/
│   │   └── sinhala/
│   │       └── Corpus.xlsx
│   │
│   └── processed/
│       ├── few_shot/
│       ├── sinhala/
│       │   └── few_shot/
│       └── multilingual/
│
├── models/
│   └── xlmr_english_sinhala_cpu_v2/
│
├── outputs/
│   ├── results/
│   └── xai/
│
├── src/
│   ├── check_dataset.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── create_few_shot.py
│   ├── train_few_shot.py
│   ├── train_setfit_few_shot.py
│   ├── train_setfit_bias_aware_64.py
│   ├── prepare_sinhala_data.py
│   ├── create_sinhala_64_shot.py
│   ├── create_bilingual_dataset.py
│   ├── train_xlmr_bilingual_cpu.py
│   ├── train_xlmr_bilingual_cpu_v2.py
│   ├── predict_xlmr_bilingual.py
│   └── predict_xlmr_bilingual_xai.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large trained models, virtual environments, caches and checkpoint directories
are excluded from Git.

## Installation

Clone the repository and open the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Preparing the Sinhala Dataset

Download `Corpus.xlsx` from the LIRNEasia Sinhala misinformation corpus and
place it in:

```text
data/raw/sinhala/Corpus.xlsx
```

Run the Sinhala preprocessing script:

```bash
python src/prepare_sinhala_data.py
```

Create the balanced Sinhala 64-shot dataset:

```bash
python src/create_sinhala_64_shot.py
```

Create the English-Sinhala bilingual datasets:

```bash
python src/create_bilingual_dataset.py
```

## Training the Bilingual Model

Run the improved CPU training script:

```bash
python src/train_xlmr_bilingual_cpu_v2.py
```

The trained model is saved locally in:

```text
models/xlmr_english_sinhala_cpu_v2
```

The evaluation results are saved in:

```text
outputs/results/xlmr_english_sinhala_cpu_v2_results.json
```

## Running Bilingual Prediction

To run prediction without LIME:

```bash
python src/predict_xlmr_bilingual.py
```

To run prediction with LIME XAI:

```bash
python src/predict_xlmr_bilingual_xai.py
```

Enter an English or Sinhala Unicode news article containing at least eight
words.

The program displays:

- Detected language
- Fake/misleading probability
- Real/credible probability
- Final prediction
- Model confidence
- LIME supporting words
- LIME opposing words

LIME uses 300 perturbed samples by default. Generating an explanation can take
several minutes when using a CPU.

Enter `Q` to close the prediction program.

## Important Interpretation

Model confidence and model accuracy are different measurements.

For example, a confidence score of 95% does not prove that an article is 95%
true. It indicates that the model strongly prefers one class based on the
language patterns it learned during training.

This model does not independently search trusted sources or verify every
factual claim.

Therefore, the system should be treated as a fake-news screening tool and not
as a complete automated fact-checking system.

## Limitations

- The Sinhala dataset contains only a small number of FALSE articles.
- The Sinhala dataset is highly imbalanced.
- PARTIAL articles were mapped to the fake/misleading class.
- The bilingual training dataset contains only 256 few-shot examples.
- Formal language, public-figure names and organisation names can influence the
  prediction.
- LIME provides model-language evidence rather than external factual evidence.
- The system currently processes text only.
- The trained XLM-R model is large and is not stored directly in the Git
  repository.

## Future Work

- Collect more manually verified Sinhala FALSE articles.
- Use a larger balanced Sinhala dataset.
- Evaluate full-data multilingual training.
- Perform probability calibration and threshold evaluation.
- Compare LIME and SHAP explanations.
- Add trusted-source evidence retrieval.
- Add claim verification and source-credibility analysis.
- Extend the system to analyse images and videos.
- Develop a web-based user interface.

## Current Research Conclusion

The project demonstrates that a single XLM-RoBERTa model can classify both
English and Sinhala news and provide human-readable LIME explanations.

The best bilingual model currently achieves 83.80% overall test accuracy and
82.93% macro F1.

The English performance is stronger than the Sinhala performance because the
Sinhala dataset contains a limited number of false articles. More balanced and
manually verified Sinhala data is required to improve the model.

The current implementation is a successful bilingual research prototype.
However, external evidence retrieval and claim verification are required before
the system can be considered suitable for real-world factual verification.