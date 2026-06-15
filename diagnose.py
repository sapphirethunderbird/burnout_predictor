"""
Diagnostic: feed a known FER2013 image directly to the saved model.

This resolves the open fork: is the model undertrained (Issue 2),
or is it the input pipeline that's broken (Issue 3)?

Interpretation:
  - If it predicts the right class → model learned something; fix the pipeline first.
  - If it predicts wrong at high confidence → model is undertrained; retrain first.

Usage: python diagnose.py
Expects archive/train/ to exist with FER2013 subdirectories.
"""

import os
import glob
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

burnout_mapping = {
    'happy': 0,
    'neutral': 1,
    'surprise': 0,
    'sad': 2,
    'angry': 2,
    'fear': 2,
    'disgust': 1,
}
class_names = {0: 'Low', 1: 'Moderate', 2: 'High'}

# Preprocessing that matches training (trainingv2.py lines 39-43)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
    model.load_state_dict(torch.load('burnout_predictor_model.pth', map_location='cpu'))
    model.eval()
    return model

def run_diagnosis(model, image_path, expected_label):
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    predicted = probs.argmax().item()
    print(f"\nImage:    {os.path.basename(image_path)}")
    print(f"Expected: {class_names[expected_label]} (class {expected_label})")
    print(f"Low:      {probs[0]:.3f}  Moderate: {probs[1]:.3f}  High: {probs[2]:.3f}")
    print(f"Predicted: {class_names[predicted]} (class {predicted})")
    correct = predicted == expected_label
    print(f"Result:   {'CORRECT ✓' if correct else 'WRONG ✗'}")
    return correct

model = load_model()
print("Model loaded. Running diagnosis on sample FER2013 images...\n")
print("=" * 60)

results = []
for emotion, burnout_class in burnout_mapping.items():
    emotion_dir = os.path.join('archive', 'train', emotion)
    if not os.path.isdir(emotion_dir):
        print(f"Skipping '{emotion}' — directory not found at {emotion_dir}")
        continue
    images = glob.glob(os.path.join(emotion_dir, '*.jpg')) + glob.glob(os.path.join(emotion_dir, '*.png'))
    if not images:
        print(f"Skipping '{emotion}' — no images found")
        continue
    correct = run_diagnosis(model, images[0], burnout_class)
    results.append(correct)

print("\n" + "=" * 60)
if results:
    accuracy = sum(results) / len(results)
    print(f"\nSample accuracy: {sum(results)}/{len(results)} correct ({accuracy:.0%})")
    print()
    if accuracy >= 0.5:
        print("VERDICT: Model learned something on clean FER2013 input.")
        print("→ The live failure is likely the input pipeline (Issue 3).")
        print("→ Next step: add face detection, align preprocessing.")
    else:
        print("VERDICT: Model cannot classify even clean FER2013 images.")
        print("→ The model itself is undertrained (Issue 2).")
        print("→ Next step: retrain on full dataset for real epochs.")
else:
    print("No images found. Check that archive/train/ contains FER2013 emotion folders.")
