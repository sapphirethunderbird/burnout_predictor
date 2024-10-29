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

# Burnout risk levels
risk_levels = {0: "Low Burnout Risk", 1: "Moderate Burnout Risk", 2: "High Burnout Risk"}

# Set up Tkinter window to display predictions
root = tk.Tk()
label_var = tk.StringVar()
label = tk.Label(root, textvariable=label_var, font=("Helvetica", 16))
label.pack()

# Timer variables for popup control
last_popup_time = 0
popup_interval = 30  # seconds

def log_high_risk():
    """Log the timestamp when high burnout risk is detected."""
    with open("high_burnout_log.csv", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp},High Burnout Risk\n")
    print(f"Logged high burnout risk at {timestamp}")

def show_high_risk_popup():
    """Show a warning popup and log if high risk is detected."""
    global last_popup_time
    current_time = time.time()
    if current_time - last_popup_time > popup_interval:
        messagebox.showwarning("High Burnout Risk", "High burnout risk detected! Please take a break.")
        log_high_risk()  # Log the high risk event
        last_popup_time = current_time

# Open webcam
cap = cv2.VideoCapture(0)

def update_tk():
    root.update_idletasks()
    root.update()
    root.after(1000, update_tk)  # Schedule next Tkinter update after 1 second

# Start the Tkinter update loop
root.after(1000, update_tk)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)

    # Run the model on the input frame
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1)

    # Map prediction to risk level
    predicted_label = risk_levels.get(predicted_idx.item(), "Unknown")
    label_var.set(f"Predicted: {predicted_label}")

    # Show popup if high risk detected and interval has passed
    if predicted_label == "High Burnout Risk":
        show_high_risk_popup()

    # Display prediction on frame
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam - Burnout Prediction", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
root.destroy()
