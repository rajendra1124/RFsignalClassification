import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt  # Added import
from tensorflow.keras.utils import plot_model  # Added import
import numpy as np


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the directory where your dataset is stored
dataset_root = 'dataset'

# Create the custom dataset
custom_dataset = CustomDataset(root_dir=dataset_root, transform=transform)

# Split the dataset into train and validation
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Define ViT model for classification
class ViTForClassification(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTForClassification, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Instantiate the ViT model for classification
num_classes = len(custom_dataset.classes)
model = ViTForClassification(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []  # Store training losses for plotting
val_losses = []    # Store validation losses for plotting

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    predictions = []  # Store predictions for confusion matrix
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())  # Append predictions to list for confusion matrix
    
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    val_accuracy = correct / total

    # Append losses to lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Plotting
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'vit_classification_model.pth')

# Print confusion matrix
true_labels = [label for _, label in val_dataset]
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)


# Compute confusion matrix percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with percentages
plt.figure(figsize=(10, 8))
plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(custom_dataset.classes))
plt.xticks(tick_marks, custom_dataset.classes, rotation=45)
plt.yticks(tick_marks, custom_dataset.classes)

# Add percentage values in each cell grid
thresh = cm_percent.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot ViT model architecture
# plot_model(model, to_file='vit_model_architecture.png', show_shapes=True)
from torchsummary import summary

# Print the model summary
summary(model, input_size=(3, 224, 224))  # Assuming input image size is (3, 224, 224)

# Save the model summary to a file
# with open('vit_model_summary.txt', 'w') as f:
#     summary(model, input_size=(3, 224, 224), print_fn=lambda x: f.write(x + '\n'))

from torchviz import make_dot

# Generate a visualization of the model architecture
x = torch.zeros((1, 3, 224, 224), dtype=torch.float, requires_grad=False)
vis_graph = make_dot(model(x), params=dict(model.named_parameters()))

# Save the visualization as an image
vis_graph.render('vit_model_architecture', format='png')


