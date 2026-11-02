"""
================================================================
  PhishGuard BERT — Day 2: Pre-trained Model Quick Test

  Uses a model ALREADY fine-tuned on phishing data from
  Hugging Face. Zero training required today.
  
  Goal: See BERT predictions working end-to-end in < 1 hour.
  
  Run: python notebooks/day2_quick_test.py
================================================================
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

print("=" * 60)
print("  PhishGuard BERT — Day 2: Quick Test")
print("=" * 60)

# ── GPU CHECK ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── LOAD PRE-TRAINED MODEL ───────────────────────────────────────
# This model was already fine-tuned on phishing emails by the
# Hugging Face community. Downloads ~440MB on first run, then cached.
print("\n[1/3] Loading pre-trained phishing detection model...")
print("  (First run downloads ~440MB — subsequent runs are instant)")

MODEL_NAME = "ealvaradob/bert-finetuned-phishing"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()  # inference mode - no gradient tracking needed

print("  ✓ Model loaded successfully")
print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── PREDICTION FUNCTION ──────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Takes an email or URL string, returns verdict + confidence.
    
    Returns:
        {
            'verdict'   : 'phishing' | 'legitimate',
            'confidence': 87.3,          # percentage 0-100
            'phish_prob': 0.873,         # raw probability
            'legit_prob': 0.127
        }
    """
    # Step 1: Tokenise — convert text to numbers BERT understands
    inputs = tokenizer(
        text,
        return_tensors='pt',     # return PyTorch tensors
        truncation=True,         # cut text longer than 512 tokens
        max_length=512,
        padding='max_length'     # pad short texts to 512 tokens
    )
    
    # Move input tensors to same device as model (GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Step 2: Run inference — no gradient calculation needed
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Step 3: Convert raw scores (logits) to probabilities
    # softmax turns e.g. [-1.2, 2.4] into [0.04, 0.96]
    probs = torch.softmax(outputs.logits, dim=1)
    probs_np = probs.cpu().numpy()[0]
    
    # Label 0 = legitimate, Label 1 = phishing
    legit_prob = float(probs_np[0])
    phish_prob = float(probs_np[1])
    
    verdict = 'phishing' if phish_prob > 0.5 else 'legitimate'
    confidence = max(phish_prob, legit_prob) * 100
    
    return {
        'verdict'   : verdict,
        'confidence': round(confidence, 1),
        'phish_prob': round(phish_prob, 4),
        'legit_prob': round(legit_prob, 4)
    }


# ── TEST CASES ───────────────────────────────────────────────────
print("\n[2/3] Running test predictions...")
print("-" * 60)

test_cases = [
    {
        "label": "PHISHING — PayPal spoof",
        "text": "URGENT: Your PayPal account has been limited. Click here immediately to verify your information or your account will be permanently suspended. http://paypa1-secure.verify-account.tk/login?token=abc123"
    },
    {
        "label": "PHISHING — Bank credential harvest",
        "text": "Dear Customer, We have detected unusual activity on your account. Please provide your full name, date of birth, and credit card number within 24 hours to avoid account closure. Reply to this email."
    },
    {
        "label": "PHISHING — Microsoft spoof",
        "text": "Security Alert: Your Microsoft account was accessed from an unknown location. Verify your identity now: http://microsoft-secure.account-verify.xyz/confirm or your account will be locked."
    },
    {
        "label": "LEGITIMATE — Work email",
        "text": "Hi team, Just a reminder that our weekly standup is tomorrow at 10am. Please have your project updates ready. The agenda has been shared in the team calendar. See you then."
    },
    {
        "label": "LEGITIMATE — GitHub notification",
        "text": "Your pull request #247 'Add phishing detection endpoint' was approved by 2 reviewers. CI checks passed. You can now merge into the main branch. View at github.com/your-repo"
    },
    {
        "label": "LEGITIMATE — Order confirmation",
        "text": "Thank you for your purchase at Amazon. Your order #112-4567890 has been confirmed. Your package will arrive by Thursday 28 November. Track your delivery on the Amazon app."
    },
    {
        "label": "EDGE CASE — Urgent but legitimate",
        "text": "IMPORTANT: Your subscription expires tomorrow. Renew now at https://www.netflix.com/account to keep watching. No action needed if you have auto-renew enabled."
    },
    {
        "label": "EDGE CASE — Phishing without links",
        "text": "Congratulations! You have been selected for a $500 gift card. To claim your reward, please reply with your full name, address, and bank account number for direct deposit."
    },
]

correct = 0
expected = [1, 1, 1, 0, 0, 0, 0, 1]  # 1=phishing, 0=legitimate

for i, (case, exp) in enumerate(zip(test_cases, expected)):
    result = predict(case['text'])
    
    predicted = 1 if result['verdict'] == 'phishing' else 0
    is_correct = predicted == exp
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    verdict_color = "PHISHING  " if result['verdict'] == 'phishing' else "LEGITIMATE"
    
    print(f"\n  {status} [{i+1}] {case['label']}")
    print(f"      Verdict    : {verdict_color}  ({result['confidence']}% confident)")
    print(f"      Phish prob : {'█' * int(result['phish_prob'] * 20):<20} {result['phish_prob']:.4f}")
    print(f"      Legit prob : {'█' * int(result['legit_prob'] * 20):<20} {result['legit_prob']:.4f}")

print("\n" + "-" * 60)
print(f"\n[3/3] Quick test accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")


# ── SPEED BENCHMARK ──────────────────────────────────────────────
import time

print("\n  Speed benchmark (10 predictions)...")
times = []
sample = test_cases[0]['text']

for _ in range(10):
    start = time.time()
    predict(sample)
    times.append(time.time() - start)

avg_ms = np.mean(times) * 1000
print(f"  Average inference time: {avg_ms:.1f}ms per email")
print(f"  Throughput: ~{1000/avg_ms:.0f} emails/second")

print("\n" + "=" * 60)
print("  Day 2 COMPLETE!")
print(f"  Pre-trained model works. Accuracy: {correct/len(test_cases)*100:.0f}%")
print("  Next: python notebooks/day3_finetune.py")
print("=" * 60)
