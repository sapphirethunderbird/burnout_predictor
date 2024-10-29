from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset

# Define burnout mapping for emotions
burnout_mapping = {
    'happy': 0,        # Low risk
    'neutral': 1,      # Moderate risk
    'surprise': 0,
    'sad': 2,          # High risk
    'angry': 2,        # High risk
    'fear': 2,         # High risk
    'disgust': 1
}

# Custom dataset class to map emotions to burnout levels
class BurnoutDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load image and original label (emotion)
        image, original_label = self.dataset[idx]
        
        # Map original label to burnout level
        class_name = self.dataset.classes[original_label]
        burnout_label = burnout_mapping.get(class_name, -1)  # -1 if no mapping found

        return image, torch.tensor(burnout_label, dtype=torch.long)

# Define the path to the training folder
data_path = 'archive/train'  # Update with the actual path

# Data transformations for ResNet input
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset and dataloader
train_data = BurnoutDataset(root_dir=data_path, transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# Check a few samples from the train_loader
for images, labels in train_loader:
    print("Image batch shape:", images.shape)  # Should be [batch_size, 3, 224, 224]
    print("Labels:", labels)                   # Should show burnout risk levels (0, 1, or 2)

    # Display a single sample label to verify correct mapping
    print("Sample label:", labels[0].item())
    break  # Remove this if you want to check the entire batch
#print("Class names:", train_data.dataset.classes)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Load a smaller model for testing (MobileNetV2)
model = models.mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 3)  # Adjust final layer for 3 burnout classes

# Use CPU only
device = torch.device("cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Preprocessing with smaller image size
preprocess = transforms.Compose([
    transforms.Resize(112),  # Reduce resolution for testing
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training loop (testing settings)
num_epochs = 1  # Try with a single epoch
max_batches = 5  # Limit batches for testing purposes

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:  # Process only max_batches for testing
            break
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/max_batches:.4f}")

print("Training complete!")
torch.save(model.state_dict(), "burnout_predictor_model.pth")

