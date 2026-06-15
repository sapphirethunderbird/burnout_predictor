import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import models, transforms


class BurnoutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Burnout Prediction and Analysis")
        self.root.geometry("800x600")

        # Define video capture and variables
        self.cap: cv2.VideoCapture | None = None
        self.model, self.preprocess, self.face_detector, self.risk_levels = self.load_model()
        self.last_popup_time = 0
        self.popup_interval = 30  # seconds

        # Create tabs
        tab_control = ttk.Notebook(self.root)
        self.feed_tab = ttk.Frame(tab_control)
        tab_control.add(self.feed_tab, text="Real-Time Feed")
        self.chart_tab = ttk.Frame(tab_control)
        tab_control.add(self.chart_tab, text="Chart Analysis")
        tab_control.pack(expand=1, fill="both")

        # Setup tabs
        self.setup_feed_tab()
        self.setup_chart_tab()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.high_risk_count = 0
        # Rolling window for tint: red when >30% of the last 60 frames are High risk.
        # This makes the display reflect burnout as frequency over time, not a single frame.
        self.recent_predictions = []
        self.rolling_window = 60

    def load_model(self):
        """Load the pre-trained MobileNetV2 model and preprocessing pipeline."""
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
        model.load_state_dict(torch.load('burnout_predictor_model.pth', map_location='cpu'))
        model.eval()
        # Canonical preprocessing — must match trainingv2.py exactly
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Haar cascade ships with OpenCV — no extra download needed
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        risk_levels = {0: "Low Burnout Risk", 1: "Moderate Burnout Risk", 2: "High Burnout Risk"}
        return model, preprocess, face_detector, risk_levels

    def setup_feed_tab(self):
        """Set up the real-time feed tab."""
        label = tk.Label(self.feed_tab, text="Click to Start Real-time Feed", font=("Arial", 16))
        label.pack(pady=20)

        start_button = tk.Button(self.feed_tab, text="Start Real-Time Feed", command=self.run_feed)
        start_button.pack(pady=10)

        self.video_label = tk.Label(self.feed_tab)
        self.video_label.pack()

        self.prediction_label = tk.Label(self.feed_tab, text="Predicted: N/A", font=("Arial", 14))
        self.prediction_label.pack(pady=10)

    def setup_chart_tab(self):
        """Set up the chart tab with a button to generate a heatmap."""
        chart_label = tk.Label(self.chart_tab, text="Chart Analysis", font=("Arial", 16))
        chart_label.pack(pady=20)

        generate_button = tk.Button(self.chart_tab, text="Generate Heatmap", command=self.generate_heatmap)
        generate_button.pack(pady=10)

        self.chart_canvas = tk.Canvas(self.chart_tab)
        self.chart_canvas.pack(expand=True, fill="both")

    def generate_heatmap(self):
        """Generate a GitHub-style heatmap from the high burnout log CSV."""
        try:
            # Load data
            df = pd.read_csv("high_burnout_log.csv", header=None, names=["timestamp", "risk_level"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            df["hour"] = df["timestamp"].dt.hour

            # Create a pivot table for the heatmap
            heatmap_data = df.pivot_table(index="hour", columns="date", aggfunc="size", fill_value=0)

            # Plot heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                heatmap_data,
                cmap="Reds",
                cbar=True,
                linewidths=0.1,
                linecolor="white",
                square=True,
                xticklabels=True,
                yticklabels=True
            )
            plt.title("High Burnout Risk Frequency", fontsize=16, weight="bold")
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Hour of Day", fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # Save and display the heatmap in the Tkinter GUI
            plt.savefig("heatmap.png", dpi=300)  # Save to a temporary file
            plt.close()

            heatmap_img = Image.open("heatmap.png").resize(
                (self.chart_canvas.winfo_width(), self.chart_canvas.winfo_height()), Image.Resampling.LANCZOS
            )

            heatmap_img = ImageTk.PhotoImage(heatmap_img)
            self.chart_canvas.create_image(0, 0, image=heatmap_img, anchor="nw")
            self.chart_canvas.image = heatmap_img  # Keep a reference to avoid garbage collection

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate heatmap: {e}")

    def run_feed(self):
        """Starts the webcam feed."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.update_feed()

    def preprocess_frame(self, frame):
        """Detect a face in the frame and preprocess it for model input.

        Returns None if no face is found — the caller skips inference for that frame.
        FER2013 was tightly-cropped faces, so we crop to match what the model trained on.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # minNeighbors=8 reduces false positives from background objects.
        # minSize is 10% of frame height so small blobs don't qualify as faces.
        min_face = int(frame.shape[0] * 0.1)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(min_face, min_face))
        if len(faces) == 0:
            return None

        # Use the largest detected face (most likely to be the user)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        return self.preprocess(pil_img).unsqueeze(0)

    def apply_red_tint(self, frame, intensity):
        """Apply a red tint to the frame based on intensity."""
        overlay = frame.copy()
        red_tint = np.full(frame.shape, (0, 0, 255), dtype=np.uint8)  # Pure red color
        intensity = np.clip(intensity, 0, 1)  # Ensure intensity stays between 0 and 1
        cv2.addWeighted(red_tint, intensity, frame, 1 - intensity, 0, overlay)
        return overlay

    def update_feed(self):
        """Continuously capture frames from the video feed, make predictions, and display them."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera. Please check your device.")
            return

        ret, frame = self.cap.read()
        if ret:
            intensity = 0
            predicted_label = "No face detected"
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is not None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    print([round(p, 3) for p in probabilities[0].tolist()]) 
                predicted_label = self.risk_levels.get(predicted_idx, "Unknown")

                # Update rolling window
                self.recent_predictions.append(predicted_idx)
                if len(self.recent_predictions) > self.rolling_window:
                    self.recent_predictions.pop(0)

                high_risk_fraction = self.recent_predictions.count(2) / len(self.recent_predictions)
                # Tint starts at 50% High and scales to full red at 80%
                intensity = min(1.0, max(0.0, (high_risk_fraction - 0.5) / 0.3))

                if predicted_label == "High Burnout Risk":
                    current_time = time.time()
                    if current_time - self.last_popup_time > self.popup_interval:
                        self.show_high_risk_popup()
                        self.last_popup_time = current_time
                    self.log_high_risk()

            # Apply tint to the frame
            tinted_frame = self.apply_red_tint(frame, intensity)

            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(tinted_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self.imgtk)

            # Update prediction label
            self.prediction_label.config(text=f"Predicted: {predicted_label}")

        # Schedule next frame update
        self.video_label.after(10, self.update_feed)


    def show_high_risk_popup(self):
        """Shows a popup if high burnout risk is detected."""
        current_time = time.time()
        if current_time - self.last_popup_time > self.popup_interval:
            messagebox.showwarning("High Burnout Risk", "High burnout risk detected! Please take a break.")
            self.last_popup_time = current_time

    def log_high_risk(self):
        """Logs high burnout risk detections."""
        self.high_risk_count += 1  # Increment the counter
        with open("high_burnout_log.csv", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp},High Burnout Risk\n")

    def on_close(self):
        """Handles app closure."""
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BurnoutApp(root)
    root.mainloop()
