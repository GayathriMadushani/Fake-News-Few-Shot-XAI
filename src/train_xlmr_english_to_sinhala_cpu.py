from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

# Configure E: cache before importing transformers.
os.environ['HF_HOME'] = r'E:\AI_Cache\huggingface'
os.environ['HF_HUB_CACHE'] = r'E:\AI_Cache\huggingface\hub'
os.environ['TORCH_HOME'] = r'E:\AI_Cache\torch'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL_NAME = 'FacebookAI/xlm-roberta-base'
DATA_DIR = Path('data/processed/multilingual')
TRAIN_PATH = DATA_DIR / 'bilingual_64_shot_train.csv'
VALIDATION_PATH = DATA_DIR / 'bilingual_validation.csv'
TEST_PATH = DATA_DIR / 'bilingual_test.csv'
OUTPUT_ROOT = Path('models/xlmr_english_to_sinhala_64shot')
ENGLISH_CHECKPOINT = OUTPUT_ROOT / 'stage1_english_best'
SINHALA_CHECKPOINT = OUTPUT_ROOT / 'stage2_sinhala_best'
RESULT_PATH = Path('outputs/results/xlmr_english_to_sinhala_64shot_results.json')

LABELS = ['fake', 'real']
LABEL_TO_ID = {'fake': 0, 'real': 1}
ID_TO_LABEL = {0: 'fake', 1: 'real'}
MAX_LENGTH = 128
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
ENGLISH_EPOCHS = 5
SINHALA_EPOCHS = 5
ENCODER_LEARNING_RATE = 2e-5
CLASSIFIER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
FROZEN_ENCODER_LAYERS = 6
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_language(path: Path, language: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'clean_text', 'label', 'language'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} missing columns: {sorted(required - set(df.columns))}')
    df = df.dropna(subset=list(required)).copy()
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    df['label'] = df['label'].str.lower()
    df['language'] = df['language'].str.lower()
    df = df[(df['language'] == language) & df['label'].isin(LABELS)
            & df['clean_text'].ne('')].reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No {language} examples in {path}')
    return df


class NewsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.labels = torch.tensor(frame['label'].map(LABEL_TO_ID).tolist(), dtype=torch.long)
        self.encodings = tokenizer(frame['clean_text'].tolist(), padding='max_length',
                                   truncation=True, max_length=MAX_LENGTH,
                                   return_tensors='pt')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'input_ids': self.encodings['input_ids'][index],
                'attention_mask': self.encodings['attention_mask'][index],
                'labels': self.labels[index]}


def freeze_lower_layers(model):
    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    for layer in model.roberta.encoder.layer[:FROZEN_ENCODER_LAYERS]:
        for p in layer.parameters():
            p.requires_grad = False


def metrics(y_true, y_pred):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average='weighted', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(macro_p), 'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1), 'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r), 'weighted_f1': float(weighted_f1),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, labels=[0, 1], target_names=LABELS,
            output_dict=True, zero_division=0),
    }


def evaluate(model, loader, device):
    model.eval()
    losses, actual, predicted = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs)
            losses.append(float(output.loss.item()))
            actual.extend(inputs['labels'].cpu().tolist())
            predicted.extend(output.logits.argmax(dim=1).cpu().tolist())
    return {'loss': float(np.mean(losses)), 'overall': metrics(actual, predicted)}


def print_metrics(title, result):
    m = result['overall']
    print(f'\n{title}: loss={result["loss"]:.4f}, accuracy={m["accuracy"]:.4f}, '
          f'macro-F1={m["macro_f1"]:.4f}')
    print('Fake precision:', round(m['classification_report']['fake']['precision'], 4))
    print('Fake recall:', round(m['classification_report']['fake']['recall'], 4))
    print('Confusion matrix [fake, real]:', m['confusion_matrix'])


def run_stage(model, tokenizer, train_df, val_df, epochs, checkpoint, stage_name, device):
    train_ds = NewsDataset(train_df, tokenizer)
    val_ds = NewsDataset(val_df, tokenizer)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    encoder_params, classifier_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (classifier_params if name.startswith('classifier.') else encoder_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': ENCODER_LEARNING_RATE},
        {'params': classifier_params, 'lr': CLASSIFIER_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_updates * WARMUP_RATIO),
        num_training_steps=total_updates)

    best_score, best_epoch, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        print(f'\n===== {stage_name}: EPOCH {epoch}/{epochs} =====', flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, 1):
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs)
            loss_sum += float(output.loss.item())
            (output.loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == len(train_loader):
                print(f'Step {step}/{len(train_loader)}, average loss={loss_sum/step:.4f}', flush=True)
        validation = evaluate(model, val_loader, device)
        print_metrics(f'{stage_name} VALIDATION epoch {epoch}', validation)
        history.append({'epoch': epoch, 'training_loss': loss_sum / len(train_loader),
                        'validation': validation})
        score = validation['overall']['macro_f1']
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            print('Saved best checkpoint:', checkpoint, flush=True)
    return {'best_epoch': best_epoch, 'best_validation_macro_f1': best_score,
            'history': history}


def main():
    set_seed(RANDOM_SEED)
    device = torch.device('cpu')
    print('Device:', device)
    print('Model:', MODEL_NAME)
    if OUTPUT_ROOT.exists() or RESULT_PATH.exists():
        raise FileExistsError(f'Output exists; refusing overwrite: {OUTPUT_ROOT} / {RESULT_PATH}')

    en_train = load_language(TRAIN_PATH, 'en')
    en_val = load_language(VALIDATION_PATH, 'en')
    si_train = load_language(TRAIN_PATH, 'si')
    si_val = load_language(VALIDATION_PATH, 'si')
    si_test = load_language(TEST_PATH, 'si')
    for name, df in [('English train', en_train), ('English validation', en_val),
                     ('Sinhala train', si_train), ('Sinhala validation', si_val),
                     ('Sinhala test', si_test)]:
        print(f'{name}: {len(df)}; {df["label"].value_counts().to_dict()}')
    if len(en_train) != 128 or len(si_train) != 128:
        raise ValueError('Expected 128 training articles per language (64/class).')
    for language, df in [('en', en_train), ('si', si_train)]:
        if df['label'].value_counts().to_dict() != {'fake': 64, 'real': 64}:
            raise ValueError(f'{language} training labels must contain 64 fake and 64 real')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID, local_files_only=True)
    freeze_lower_layers(model)
    model.to(device)
    print('Frozen encoder layers:', FROZEN_ENCODER_LAYERS)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    start = time.time()

    # Stage 1: English only; select checkpoint using English validation.
    english_results = run_stage(model, tokenizer, en_train, en_val, ENGLISH_EPOCHS,
                                ENGLISH_CHECKPOINT, 'ENGLISH', device)
    del model
    model = AutoModelForSequenceClassification.from_pretrained(
        ENGLISH_CHECKPOINT, local_files_only=True).to(device)
    freeze_lower_layers(model)

    # Stage 2: Sinhala only; select checkpoint using Sinhala validation.
    sinhala_results = run_stage(model, tokenizer, si_train, si_val, SINHALA_EPOCHS,
                                SINHALA_CHECKPOINT, 'SINHALA', device)
    del model
    model = AutoModelForSequenceClassification.from_pretrained(
        SINHALA_CHECKPOINT, local_files_only=True).to(device)
    test_loader = DataLoader(NewsDataset(si_test, tokenizer), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    final_test = evaluate(model, test_loader, device)
    print_metrics('FINAL SINHALA TEST', final_test)

    results = {
        'model_name': MODEL_NAME,
        'experiment': 'English to Sinhala sequential 64-shot per class transfer',
        'seed': RANDOM_SEED, 'device': str(device),
        'english_training_samples': len(en_train),
        'english_validation_samples': len(en_val),
        'sinhala_training_samples': len(si_train),
        'sinhala_validation_samples': len(si_val),
        'sinhala_test_samples': len(si_test),
        'maximum_length': MAX_LENGTH, 'batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'english_epochs': ENGLISH_EPOCHS, 'sinhala_epochs': SINHALA_EPOCHS,
        'encoder_learning_rate': ENCODER_LEARNING_RATE,
        'classifier_learning_rate': CLASSIFIER_LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'warmup_ratio': WARMUP_RATIO,
        'frozen_encoder_layers': FROZEN_ENCODER_LAYERS,
        'english_stage': english_results, 'sinhala_stage': sinhala_results,
        'training_minutes': (time.time() - start) / 60,
        'test': final_test,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\nTraining completed successfully.')
    print('Stage 1 model:', ENGLISH_CHECKPOINT)
    print('Stage 2 model:', SINHALA_CHECKPOINT)
    print('Results:', RESULT_PATH)


if __name__ == '__main__':
    main()
