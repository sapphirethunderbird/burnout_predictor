import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import messagebox
import time
from datetime import datetime

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(weights='DEFAULT')
num_classes = 3
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.eval()

# Preprocessing function for MobileNetV2
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
