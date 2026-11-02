"""
================================================================
  PhishGuard BERT — Day 6: Evaluation & Metrics Dashboard

  Generates all the charts and metrics you need for your
  capstone report and live demo:
    - Confusion matrix
    - ROC curve
    - Precision-Recall curve
    - Training loss curve
    - Confidence distribution
    - Per-class metrics table

  Run: python notebooks/day6_evaluation.py
  Output: models/evaluation_report.png  (include in your report)
================================================================
"""

import torch
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from torch.utils.data import Dataset, DataLoader

# ── STYLE ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#050a0f',
    'axes.facecolor'   : '#0a1520',
    'axes.edgecolor'   : '#1a3a52',
    'axes.labelcolor'  : '#c8dde8',
    'text.color'       : '#c8dde8',
    'xtick.color'      : '#6a9ab8',
    'ytick.color'      : '#6a9ab8',
    'grid.color'       : '#0e2538',
    'grid.alpha'       : 0.5,
    'font.family'      : 'monospace',
    'font.size'        : 10,
})

ACCENT  = '#00d4ff'
DANGER  = '#ff3b5c'
SAFE    = '#00e887'
WARN    = '#ffb347'
PURPLE  = '#a78bfa'

print("=" * 60)
print("  PhishGuard BERT — Day 6: Evaluation Dashboard")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── LOAD MODEL ───────────────────────────────────────────────────
MODEL_PATH = './models/phishguard-bert-finetuned'
FALLBACK   = 'ealvaradob/bert-finetuned-phishing'

model_source = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK
print(f"\n  Loading: {model_source}")

tokenizer = AutoTokenizer.from_pretrained(model_source)
model     = AutoModelForSequenceClassification.from_pretrained(model_source)
model     = model.to(device)
model.eval()
print("  ✓ Model loaded")

# ── LOAD TEST DATA ───────────────────────────────────────────────
if os.path.exists('data/test.csv'):
    df_test = pd.read_csv('data/test.csv')
    print(f"  ✓ Test set: {len(df_test)} samples")
else:
    # Minimal fallback test set
    print("  ! No test.csv found — using built-in test set")
    test_data = {
        'content': [
            "URGENT: Verify your PayPal account now or face suspension http://paypa1.verify.tk",
            "Your Amazon order #123 has shipped. Track at amazon.com",
            "Dear customer, provide your SSN and credit card to restore access",
            "Hi team, meeting at 10am tomorrow. Agenda in calendar.",
            "FINAL NOTICE: Your account will be closed in 24 hours. Click here.",
            "Your GitHub PR was approved. You can now merge into main.",
            "Congratulations! You won $1,000,000. Send bank details to claim.",
            "Invoice attached for your records. Payment due in 30 days.",
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df_test = pd.DataFrame(test_data)


# ── INFERENCE ON TEST SET ────────────────────────────────────────
print("\n  Running inference on test set...")

all_labels, all_preds, all_probs = [], [], []

batch_size = 16
for i in range(0, len(df_test), batch_size):
    batch_df = df_test.iloc[i:i+batch_size]
    texts    = batch_df['content'].str[:512].tolist()
    labels   = batch_df['label'].tolist()

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    all_labels.extend(labels)
    all_preds.extend(preds.tolist())
    all_probs.extend(probs[:, 1].tolist())  # phishing probability

    progress = min((i + batch_size) / len(df_test), 1.0)
    print(f"  {'█' * int(progress * 40):<40} {progress*100:.0f}%", end='\r')

print(f"\n  ✓ Inference complete on {len(all_labels)} samples")

# ── COMPUTE METRICS ──────────────────────────────────────────────
accuracy  = accuracy_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds, weighted=True)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
cm        = confusion_matrix(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)
prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)

print(f"\n  Accuracy  : {accuracy*100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")


# ── PLOT DASHBOARD ───────────────────────────────────────────────
print("\n  Generating evaluation dashboard...")

fig = plt.figure(figsize=(18, 12), facecolor='#050a0f')
fig.suptitle('PhishGuard BERT — Evaluation Dashboard', 
             fontsize=18, color=ACCENT, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)


# ── PLOT 1: Confusion Matrix ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax1,
    cmap=sns.light_palette(ACCENT, as_cmap=True),
    linewidths=1, linecolor='#0e2538',
    xticklabels=['Legitimate', 'Phishing'],
    yticklabels=['Legitimate', 'Phishing'],
    cbar_kws={'shrink': 0.8}
)
ax1.set_title('Confusion Matrix', color=ACCENT, pad=12)
ax1.set_xlabel('Predicted', labelpad=8)
ax1.set_ylabel('Actual', labelpad=8)

# Annotate TP, TN, FP, FN
for i in range(2):
    for j in range(2):
        label = ['TN','FP','FN','TP'][i*2+j]
        color = SAFE if label in ('TP','TN') else DANGER
        ax1.text(j+0.5, i+0.75, label, ha='center', va='center',
                 color=color, fontsize=9, alpha=0.7)


# ── PLOT 2: ROC Curve ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color=ACCENT, lw=2, label=f'BERT (AUC = {roc_auc:.3f})')
ax2.plot([0,1],[0,1], color='#3a6a88', lw=1, linestyle='--', label='Random (AUC = 0.500)')
ax2.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', color=ACCENT, pad=12)
ax2.legend(loc='lower right', framealpha=0.2)
ax2.grid(True, alpha=0.3)
ax2.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', color=ACCENT,
         fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#0a1520', edgecolor=ACCENT, alpha=0.8))


# ── PLOT 3: Precision-Recall Curve ──────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(rec_curve, prec_curve, color=PURPLE, lw=2)
ax3.fill_between(rec_curve, prec_curve, alpha=0.1, color=PURPLE)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', color=ACCENT, pad=12)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1.02])
ax3.grid(True, alpha=0.3)
pr_auc = auc(rec_curve, prec_curve)
ax3.text(0.05, 0.1, f'PR-AUC = {pr_auc:.4f}', color=PURPLE,
         fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#0a1520', edgecolor=PURPLE, alpha=0.8))


# ── PLOT 4: Training Loss (if history exists) ────────────────────
ax4 = fig.add_subplot(gs[1, 0])
history_path = 'models/training_history.json'
if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)['history']
    epochs = range(1, len(history['train_loss']) + 1)
    ax4.plot(epochs, history['train_loss'], color=DANGER,  lw=2, marker='o', label='Train Loss')
    ax4.plot(epochs, [1-v for v in history['val_acc']],  color=WARN, lw=2, marker='s', linestyle='--', label='Val Error')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss / Error')
    ax4.set_title('Training History', color=ACCENT, pad=12)
    ax4.legend(framealpha=0.2)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Run day3_finetune.py\nto generate training history',
             ha='center', va='center', color='#3a6a88', fontsize=11,
             transform=ax4.transAxes)
    ax4.set_title('Training History', color=ACCENT, pad=12)


# ── PLOT 5: Confidence Distribution ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
phish_confs = [p for p, l in zip(all_probs, all_labels) if l == 1]
legit_confs = [p for p, l in zip(all_probs, all_labels) if l == 0]
bins = np.linspace(0, 1, 25)
ax5.hist(legit_confs, bins=bins, color=SAFE,   alpha=0.7, label='Legitimate', density=True)
ax5.hist(phish_confs, bins=bins, color=DANGER, alpha=0.7, label='Phishing',   density=True)
ax5.axvline(x=0.5, color=WARN, linestyle='--', lw=1.5, label='Decision threshold')
ax5.set_xlabel('Phishing Probability')
ax5.set_ylabel('Density')
ax5.set_title('Confidence Distribution', color=ACCENT, pad=12)
ax5.legend(framealpha=0.2)
ax5.grid(True, alpha=0.3)


# ── PLOT 6: Metrics Summary Card ─────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

metrics_data = [
    ('Accuracy',          f'{accuracy*100:.2f}%',  SAFE   if accuracy > 0.9 else WARN),
    ('F1 Score',          f'{f1:.4f}',             SAFE   if f1 > 0.9       else WARN),
    ('Precision',         f'{precision:.4f}',      SAFE   if precision > 0.9 else WARN),
    ('Recall',            f'{recall:.4f}',         SAFE   if recall > 0.9   else WARN),
    ('ROC-AUC',           f'{roc_auc:.4f}',        SAFE   if roc_auc > 0.95 else WARN),
    ('PR-AUC',            f'{pr_auc:.4f}',         SAFE   if pr_auc > 0.9   else WARN),
    ('Test Samples',      f'{len(all_labels)}',    ACCENT),
    ('Phishing Rate',     f'{sum(all_labels)/len(all_labels)*100:.1f}%', ACCENT),
]

ax6.set_title('Evaluation Summary', color=ACCENT, pad=12)
y_pos = 0.95
for name, value, color in metrics_data:
    ax6.text(0.05, y_pos, name,  transform=ax6.transAxes, color='#6a9ab8', fontsize=11)
    ax6.text(0.65, y_pos, value, transform=ax6.transAxes, color=color,     fontsize=12, fontweight='bold')
    ax6.axhline(y=y_pos - 0.02, xmin=0.03, xmax=0.97,
                color='#0e2538', linewidth=0.5, transform=ax6.transAxes)
    y_pos -= 0.105

# Grade badge
grade = 'A+' if accuracy > 0.95 else 'A' if accuracy > 0.90 else 'B' if accuracy > 0.85 else 'C'
grade_color = SAFE if grade in ('A+','A') else WARN if grade == 'B' else DANGER
ax6.text(0.5, 0.08, grade, transform=ax6.transAxes,
         color=grade_color, fontsize=36, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a1520', edgecolor=grade_color))


# ── SAVE ─────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
output_path = 'models/evaluation_report.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#050a0f')
print(f"\n  ✓ Dashboard saved to: {output_path}")

# Also save metrics as JSON for the capstone report
metrics_json = {
    'accuracy' : round(accuracy, 4),
    'f1'       : round(f1, 4),
    'precision': round(precision, 4),
    'recall'   : round(recall, 4),
    'roc_auc'  : round(roc_auc, 4),
    'pr_auc'   : round(pr_auc, 4),
    'test_samples': len(all_labels),
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(
        all_labels, all_preds,
        target_names=['Legitimate', 'Phishing'],
        output_dict=True
    )
}

with open('models/final_metrics.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)

print("  ✓ Metrics saved to: models/final_metrics.json")
print("\n  Classification Report:")
print(classification_report(all_labels, all_preds,
      target_names=['Legitimate', 'Phishing'], digits=4))

print("=" * 60)
print("  Day 6 COMPLETE!")
print("  evaluation_report.png → include in your capstone report")
print("  Next: Day 7 — final polish")
print("=" * 60)
