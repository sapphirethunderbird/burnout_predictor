import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split

burnout_mapping = {
    'happy': 0,
    'neutral': 1,
    'surprise': 0,
    'sad': 2,
    'angry': 2,
    'fear': 2,
    'disgust': 1,
}

class BurnoutDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, original_label = self.dataset[idx]
        class_name = self.dataset.classes[original_label]
        burnout_label = burnout_mapping.get(class_name, -1)
        return image, torch.tensor(burnout_label, dtype=torch.long)


# Canonical preprocessing — must match inference in gui.py
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = BurnoutDataset(root_dir='archive/train', transform=preprocess)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(f"Training samples: {train_size}  |  Validation samples: {val_size}")

# Load existing model (backbone was originally initialized from ImageNet weights)
# — avoids any network download, and gives us a better starting point than random init
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load('burnout_predictor_model.pth', map_location='cpu'))

device = torch.device('cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # --- Training pass ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation pass ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}]  loss: {avg_train_loss:.4f}  val_acc: {val_acc:.3f}")

print("\nTraining complete!")
torch.save(model.state_dict(), 'burnout_predictor_model.pth')
print("Saved to burnout_predictor_model.pth")
