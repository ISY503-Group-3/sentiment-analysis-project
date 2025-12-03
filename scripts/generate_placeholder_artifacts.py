#!/usr/bin/env python3
"""Generate small placeholder evaluation PNGs for the Streamlit UI.

Creates:
- artifacts/evaluation_confusion_matrix.png
- artifacts/evaluation_roc_curve.png
- artifacts/evaluation_confidence_hist.png
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

out = Path(__file__).parent.parent / 'artifacts'
out.mkdir(parents=True, exist_ok=True)

# Confusion matrix placeholder
cm = np.array([[50, 10], [8, 32]])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (placeholder)')
plt.tight_layout()
plt.savefig(out / 'evaluation_confusion_matrix.png', dpi=150)
plt.close()

# ROC curve placeholder
fpr = np.linspace(0,1,100)
tpr = np.sqrt(1 - (1 - fpr)**2)  # some curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = 0.85)')
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (placeholder)')
plt.legend()
plt.tight_layout()
plt.savefig(out / 'evaluation_roc_curve.png', dpi=150)
plt.close()

# Confidence histogram placeholder
probs = np.concatenate([np.random.beta(2,5,500), np.random.beta(5,2,300)])
plt.figure(figsize=(6,4))
plt.hist(probs, bins=25, color='slateblue', alpha=0.85)
plt.title('Model Prediction Confidence Distribution (placeholder)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(out / 'evaluation_confidence_hist.png', dpi=150)
plt.close()

print('Wrote placeholder images to', out)
