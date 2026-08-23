"""
Evaluate the trained model on the held-out FER2013 test set and plot
confusion matrices.

Two matrices are produced, because the model does NOT predict emotions —
it predicts three burnout risk classes derived from emotion labels
(see burnout_mapping in trainingv2.py):

  1. Burnout 3x3  — true burnout class vs. predicted burnout class.
                    This is the model's actual task.
  2. Emotion 7x3  — original FER2013 emotion vs. predicted burnout class.
                    This shows *which emotions* get misrouted, e.g. whether
                    'disgust' (mapped to Moderate) is really being pulled
                    toward High along with 'angry' and 'fear'.

Usage: python confusion_matrix.py
Expects archive/test/ with FER2013 emotion subdirectories and
burnout_predictor_model.pth in the project root.
Writes confusion_matrix.png.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Same mapping as trainingv2.py — keep in sync
burnout_mapping = {
    'happy': 0,
    'neutral': 1,
    'surprise': 0,
    'sad': 2,
    'angry': 2,
    'fear': 2,
    'disgust': 1,
}
risk_names = ['Low', 'Moderate', 'High']

# Canonical preprocessing — must match trainingv2.py and gui.py exactly
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root='archive/test', transform=preprocess)
loader = DataLoader(dataset, batch_size=64, shuffle=False)
emotion_names = dataset.classes
print(f"Test images: {len(dataset)}  |  emotions: {emotion_names}")

device = torch.device('cpu')
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load('burnout_predictor_model.pth', map_location='cpu'))
model = model.to(device).eval()

# emotion_index -> burnout class, aligned with ImageFolder's class ordering
emotion_to_risk = np.array([burnout_mapping[e] for e in emotion_names])

all_preds, all_emotions = [], []
with torch.no_grad():
    for i, (images, emotion_labels) in enumerate(loader):
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_emotions.append(emotion_labels)
        print(f"\rBatch {i+1}/{len(loader)}", end='', flush=True)
print()

preds = torch.cat(all_preds).numpy()
emotions = torch.cat(all_emotions).numpy()
true_risk = emotion_to_risk[emotions]

# --- Matrix 1: burnout class (3x3) ---
cm_risk = np.zeros((3, 3), dtype=int)
for t, p in zip(true_risk, preds):
    cm_risk[t, p] += 1

# --- Matrix 2: emotion (7) vs predicted risk (3) ---
cm_emotion = np.zeros((len(emotion_names), 3), dtype=int)
for e, p in zip(emotions, preds):
    cm_emotion[e, p] += 1

accuracy = (preds == true_risk).mean()
print(f"\nOverall test accuracy (burnout classes): {accuracy:.3f}")
print("\nPer-class recall:")
for i, name in enumerate(risk_names):
    support = cm_risk[i].sum()
    recall = cm_risk[i, i] / support if support else float('nan')
    print(f"  {name:<9} {recall:.3f}  (n={support})")

print("\nPer-emotion breakdown (row-normalized, predicted risk):")
for i, name in enumerate(emotion_names):
    row = cm_emotion[i]
    frac = row / row.sum()
    target = risk_names[emotion_to_risk[i]]
    print(f"  {name:<9} expected {target:<9} "
          f"Low {frac[0]:.2f}  Moderate {frac[1]:.2f}  High {frac[2]:.2f}  (n={row.sum()})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_risk, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=risk_names, yticklabels=risk_names, ax=axes[0])
axes[0].set_title(f'Burnout risk confusion matrix\ntest accuracy = {accuracy:.1%}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True (from emotion label)')

cm_emotion_norm = cm_emotion / cm_emotion.sum(axis=1, keepdims=True)
annot = np.array([[f"{cm_emotion[i, j]}\n{cm_emotion_norm[i, j]:.0%}"
                   for j in range(3)] for i in range(len(emotion_names))])
sns.heatmap(cm_emotion_norm, annot=annot, fmt='', cmap='Blues', cbar=False,
            vmin=0, vmax=1, xticklabels=risk_names,
            yticklabels=[f"{e}\n→{risk_names[emotion_to_risk[i]]}"
                         for i, e in enumerate(emotion_names)], ax=axes[1])
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
axes[1].set_title('FER2013 emotion vs. predicted burnout risk\n(row-normalized)')
axes[1].set_xlabel('Predicted risk')
axes[1].set_ylabel('True emotion → expected risk')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nSaved confusion_matrix.png")
