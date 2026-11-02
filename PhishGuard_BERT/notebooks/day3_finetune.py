"""
================================================================
  PhishGuard BERT — Day 3: Fine-tuning on Your Dataset

  Takes the pre-trained BERT model and continues training it
  on YOUR phishing dataset from Day 1. This customises it
  for the exact kind of phishing emails you care about.

  Expected time: 20–60 min depending on GPU + dataset size.

  Run: python notebooks/day3_finetune.py
================================================================
"""

import torch
import pandas as pd
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("=" * 60)
print("  PhishGuard BERT — Day 3: Fine-tuning")
print("=" * 60)

# ── CONFIG ───────────────────────────────────────────────────────
# Tweak these if you run out of VRAM or want faster training
CONFIG = {
    'model_name'    : 'bert-base-uncased',  # base BERT to fine-tune
    'max_length'    : 256,     # token limit (256 = faster, 512 = more accurate)
    'batch_size'    : 16,      # reduce to 8 if you get CUDA out of memory error
    'num_epochs'    : 3,       # 3 epochs is usually enough for fine-tuning
    'learning_rate' : 2e-5,    # standard for BERT fine-tuning
    'warmup_ratio'  : 0.1,     # 10% of steps for LR warmup
    'output_dir'    : './models/phishguard-bert-finetuned',
    'seed'          : 42,
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device : {device}")
if device.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM   : {vram:.1f} GB")
    if vram < 6:
        print("  ⚠ < 6GB VRAM — reducing batch size to 8")
        CONFIG['batch_size'] = 8

print(f"\n  Config:")
for k, v in CONFIG.items():
    print(f"    {k:<18}: {v}")


# ── 1. DATASET CLASS ─────────────────────────────────────────────
class PhishingDataset(Dataset):
    """
    PyTorch Dataset — wraps our CSV data for the DataLoader.
    The DataLoader uses this to feed batches to the model.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts  = dataframe['content'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        # How many samples in this dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Called once per sample — tokenise on the fly
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids'     : encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label'         : torch.tensor(label, dtype=torch.long)
        }


# ── 2. LOAD DATA ─────────────────────────────────────────────────
print("\n[1/5] Loading datasets...")

# Check data files exist
for split in ['train', 'validation', 'test']:
    path = f'data/{split}.csv'
    if not os.path.exists(path):
        print(f"  ! {path} not found. Run data/prepare_dataset.py first.")
        exit(1)

df_train = pd.read_csv('data/train.csv')
df_val   = pd.read_csv('data/validation.csv')
df_test  = pd.read_csv('data/test.csv')

print(f"  ✓ Train      : {len(df_train)} samples")
print(f"  ✓ Validation : {len(df_val)} samples")
print(f"  ✓ Test       : {len(df_test)} samples")
print(f"  ✓ Train phishing rate: {df_train['label'].mean()*100:.1f}%")


# ── 3. LOAD MODEL & TOKENIZER ────────────────────────────────────
print(f"\n[2/5] Loading {CONFIG['model_name']}...")
print("  (Downloads ~440MB on first run)")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=2  # binary: phishing or legitimate
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ✓ Total parameters    : {total_params:,}")
print(f"  ✓ Trainable parameters: {trainable:,}")


# ── 4. CREATE DATALOADERS ────────────────────────────────────────
print("\n[3/5] Creating data loaders...")

train_dataset = PhishingDataset(df_train, tokenizer, CONFIG['max_length'])
val_dataset   = PhishingDataset(df_val,   tokenizer, CONFIG['max_length'])
test_dataset  = PhishingDataset(df_test,  tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'] * 2)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'] * 2)

print(f"  ✓ Train batches     : {len(train_loader)}")
print(f"  ✓ Validation batches: {len(val_loader)}")


# ── 5. OPTIMIZER & SCHEDULER ────────────────────────────────────
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01  # L2 regularisation to prevent overfitting
)

total_steps  = len(train_loader) * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

# Linear warmup then linear decay — standard for BERT fine-tuning
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\n  Training steps  : {total_steps}")
print(f"  Warmup steps    : {warmup_steps}")


# ── 6. EVALUATION FUNCTION ──────────────────────────────────────
def evaluate(model, loader, split_name="Validation"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, all_preds, all_labels


# ── 7. TRAINING LOOP ────────────────────────────────────────────
print(f"\n[4/5] Training for {CONFIG['num_epochs']} epochs...")
print("=" * 60)

best_val_f1   = 0
history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

for epoch in range(CONFIG['num_epochs']):
    # ── TRAIN ──
    model.train()
    total_loss   = 0
    epoch_start  = time.time()
    
    print(f"\n  Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"  {'─'*50}")

    for step, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels       # passing labels computes loss inside model
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass — compute gradients
        loss.backward()

        # Clip gradients — prevents exploding gradients in transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Progress update every 20 steps
        if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
            elapsed = time.time() - epoch_start
            pct     = (step + 1) / len(train_loader)
            eta     = elapsed / pct * (1 - pct)
            avg_loss = total_loss / (step + 1)
            bar     = '█' * int(pct * 30) + '░' * (30 - int(pct * 30))
            print(f"  [{bar}] {pct*100:5.1f}%  loss: {avg_loss:.4f}  ETA: {eta:.0f}s", end='\r')

    avg_train_loss = total_loss / len(train_loader)
    
    # ── VALIDATE ──
    val_acc, val_f1, _, _ = evaluate(model, val_loader)
    
    epoch_time = time.time() - epoch_start
    print(f"\n  Epoch {epoch+1} complete ({epoch_time:.0f}s)")
    print(f"    Train Loss : {avg_train_loss:.4f}")
    print(f"    Val Acc    : {val_acc*100:.2f}%")
    print(f"    Val F1     : {val_f1:.4f}")

    history['train_loss'].append(avg_train_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        model.save_pretrained(CONFIG['output_dir'])
        tokenizer.save_pretrained(CONFIG['output_dir'])
        print(f"    ✓ New best model saved! (F1: {val_f1:.4f})")


# ── 8. FINAL TEST EVALUATION ────────────────────────────────────
print("\n[5/5] Loading best model for final test evaluation...")

best_model = AutoModelForSequenceClassification.from_pretrained(CONFIG['output_dir'])
best_model = best_model.to(device)

test_acc, test_f1, test_preds, test_labels = evaluate(best_model, test_loader, "Test")

print("\n" + "=" * 60)
print("  FINAL TEST SET RESULTS")
print("=" * 60)
print(f"\n  Accuracy  : {test_acc*100:.2f}%")
print(f"  F1 Score  : {test_f1:.4f}")
print(f"\n  Detailed report:")
print(classification_report(
    test_labels, test_preds,
    target_names=['Legitimate', 'Phishing'],
    digits=4
))

# Save training history for Day 6 metrics dashboard
import json
with open('models/training_history.json', 'w') as f:
    json.dump({
        'history': history,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'config': {k: str(v) for k, v in CONFIG.items()}
    }, f, indent=2)

print("\n  ✓ Training history saved to models/training_history.json")
print("\n" + "=" * 60)
print("  Day 3 COMPLETE!")
print(f"  Your fine-tuned model is at: {CONFIG['output_dir']}")
print("  Next: python backend/app.py")
print("=" * 60)
