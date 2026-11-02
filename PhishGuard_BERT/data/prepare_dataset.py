"""
================================================================
  PhishGuard BERT — Day 1: Dataset Preparation
  
  This script downloads and combines phishing email datasets
  from Hugging Face (no manual CSV downloads needed).
  
  Run: python data/prepare_dataset.py
================================================================
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os

print("=" * 60)
print("  PhishGuard BERT — Dataset Preparation")
print("=" * 60)

# ── OPTION A: Download from Hugging Face (recommended) ──────────
# This dataset has 18,650 emails, pre-labelled: 0=legit, 1=phishing
print("\n[1/4] Downloading phishing dataset from Hugging Face...")

try:
    dataset = load_dataset("ealvaradob/phishing-dataset", "emails")
    df_train = dataset['train'].to_pandas()
    
    # Rename columns to standard names
    # The dataset has: 'text' and 'label' (0=legit, 1=phishing)
    df_train = df_train.rename(columns={'text': 'content', 'label': 'label'})
    print(f"  ✓ Downloaded {len(df_train)} emails")

except Exception as e:
    print(f"  ! Hugging Face download failed: {e}")
    print("  → Falling back to manual CSV creation...")
    
    # ── OPTION B: Manual CSV fallback ───────────────────────────
    # If Hugging Face fails, use this small sample to test the pipeline
    sample_data = {
        'content': [
            # Phishing examples (label=1)
            "URGENT: Your PayPal account has been limited. Click here immediately to verify your information or your account will be permanently suspended. http://paypa1-secure.verify.tk/login",
            "Dear Customer, We detected suspicious activity on your bank account. Please provide your full name, date of birth, SSN, and credit card number to restore access.",
            "Congratulations! You have been selected as a winner. Claim your $1,000,000 prize NOW. Send your banking details to collect your reward.",
            "Your Amazon account requires immediate verification. Failure to verify within 24 hours will result in account closure. Click: http://amaz0n-verify.suspicious.ru",
            "FINAL NOTICE: Your email storage is full. Click here to upgrade immediately or lose all emails. This is your last warning.",
            "Security Alert: Unauthorized login detected from Russia. Verify your identity NOW: http://microsoft-secure.verify-account.xyz/login",
            "Dear valued member, Your account has been compromised. Update your password immediately using this link to prevent unauthorized access.",
            "You have 1 unread security notification. Your Apple ID has been used to sign in on a new device. Verify at: http://apple-id.account-verify.com",
            
            # Legitimate examples (label=0)
            "Hi team, Just a reminder that our weekly standup is at 10am tomorrow. Please have your updates ready. Looking forward to seeing everyone.",
            "Your GitHub pull request #142 has been approved. The CI checks passed successfully. You can now merge into main.",
            "Invoice #INV-2024-089 is attached for your records. Payment is due within 30 days. Please contact accounts@company.com with any questions.",
            "Newsletter: This week in technology — AI advances in healthcare, new smartphone releases, and climate tech funding rounds.",
            "Your package has been dispatched and will arrive by Thursday. Track your order at: https://www.fedex.com/tracking/12345",
            "Meeting notes from today's product review have been shared to the team Drive folder. Action items are highlighted in yellow.",
            "Thank you for your recent purchase. Your receipt is attached. For returns, visit our official website within 30 days.",
            "Your monthly statement is now available. Log in to your account at https://www.barclays.co.uk to view your transactions.",
        ],
        'label': [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]
    }
    df_train = pd.DataFrame(sample_data)
    print(f"  ✓ Created sample dataset with {len(df_train)} emails")

# ── DATA EXPLORATION ─────────────────────────────────────────────
print("\n[2/4] Dataset overview:")
print(f"  Total samples   : {len(df_train)}")
print(f"  Phishing (1)    : {df_train['label'].sum()} ({df_train['label'].mean()*100:.1f}%)")
print(f"  Legitimate (0)  : {(df_train['label']==0).sum()} ({(df_train['label']==0).mean()*100:.1f}%)")
print(f"  Avg text length : {df_train['content'].str.len().mean():.0f} characters")
print(f"  Max text length : {df_train['content'].str.len().max()} characters")

# ── DATA CLEANING ────────────────────────────────────────────────
print("\n[3/4] Cleaning data...")

# Remove empty rows
original_len = len(df_train)
df_train = df_train.dropna(subset=['content', 'label'])
df_train = df_train[df_train['content'].str.strip() != '']
df_train['content'] = df_train['content'].str.strip()

# Truncate very long emails (BERT max is 512 tokens ≈ ~400 words)
# We keep the first 1000 characters which is usually enough
df_train['content'] = df_train['content'].str[:1000]

# Ensure label is integer
df_train['label'] = df_train['label'].astype(int)

print(f"  ✓ Removed {original_len - len(df_train)} invalid rows")
print(f"  ✓ Final dataset: {len(df_train)} emails")

# ── TRAIN / VALIDATION / TEST SPLIT ─────────────────────────────
print("\n[4/4] Creating train/validation/test splits...")

# Shuffle the data
df_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Split: 70% train, 15% validation, 15% test
n = len(df_shuffled)
train_end  = int(n * 0.70)
val_end    = int(n * 0.85)

df_tr   = df_shuffled.iloc[:train_end]
df_val  = df_shuffled.iloc[train_end:val_end]
df_test = df_shuffled.iloc[val_end:]

# Save to CSV files
os.makedirs('data', exist_ok=True)
df_tr.to_csv('data/train.csv',      index=False)
df_val.to_csv('data/validation.csv', index=False)
df_test.to_csv('data/test.csv',     index=False)
df_shuffled.to_csv('data/full.csv', index=False)

print(f"  ✓ Train set      : {len(df_tr)} samples → data/train.csv")
print(f"  ✓ Validation set : {len(df_val)} samples → data/validation.csv")
print(f"  ✓ Test set       : {len(df_test)} samples → data/test.csv")

print("\n" + "=" * 60)
print("  Day 1 COMPLETE!")
print("  Next: python notebooks/day2_quick_test.py")
print("=" * 60)
