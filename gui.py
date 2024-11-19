import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BurnoutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Burnout Prediction and Analysis")
        self.root.geometry("800x600")

        # Define video capture and variables
        self.cap: cv2.VideoCapture | None = None
        self.model, self.preprocess, self.risk_levels = self.load_model()
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

    def load_model(self):
        """Load the pre-trained MobileNetV2 model and preprocessing pipeline."""
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(model.last_channel, 3)  # Three risk levels
        model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        risk_levels = {0: "Low Burnout Risk", 1: "Moderate Burnout Risk", 2: "High Burnout Risk"}
        return model, preprocess, risk_levels

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

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

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
        """Preprocess a video frame for model input."""
        try:
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply preprocessing directly to the NumPy array
            input_tensor = self.preprocess(frame_rgb)
            return input_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error in preprocessing frame: {e}")
            return None



    def update_feed(self):
        """Continuously capture frames from the video feed, make predictions, and display them."""
        # Initialize the camera if not already done
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        # Confirm the camera is initialized and accessible
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera. Please check your device.")
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.imgtk = ImageTk.PhotoImage(image=img)  # Prevent garbage collection
            self.video_label.configure(image=self.imgtk)

            # Preprocess the frame for the model
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is not None:
                # Run the model prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()

                # Map the prediction to a risk level
                predicted_label = self.risk_levels.get(predicted_idx, "Unknown")

                # Log high burnout risk predictions and show a popup
                if predicted_label == "High Burnout Risk":
                    current_time = time.time()
                    if current_time - self.last_popup_time > self.popup_interval:
                        self.show_high_risk_popup()
                        self.last_popup_time = current_time

                    # Log to CSV regardless of popup timing
                    self.log_high_risk()

                # Display prediction on the video feed
                self.prediction_label.config(text=f"Predicted: {predicted_label}")

        # Schedule the next frame update
        self.video_label.after(10, self.update_feed)

    def show_high_risk_popup(self):
        """Shows a popup if high burnout risk is detected."""
        current_time = time.time()
        if current_time - self.last_popup_time > self.popup_interval:
            messagebox.showwarning("High Burnout Risk", "High burnout risk detected! Please take a break.")
            self.log_high_risk()
            self.last_popup_time = current_time

    def log_high_risk(self):
        """Logs high burnout risk detections."""
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
