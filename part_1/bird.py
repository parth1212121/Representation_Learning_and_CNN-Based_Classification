
import sys
import os
import csv
import torch
import random
import math
import time
import PIL
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchvision.transforms import functional as F_transforms
from PIL import Image
from collections import Counter
import copy


# Verify command-line arguments
if len(sys.argv) < 4:
    print("Usage:")
    print("Training: python bird.py path_to_dataset train bird.pth")
    print("Testing: python bird.py path_to_dataset test bird.pth")
    sys.exit(1)

# Parse command-line arguments
data_path = sys.argv[1]
mode = sys.argv[2]
model_path = sys.argv[3]


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Calculate class weights based on training data distribution
def calculate_class_weights(dataset):
    label_counts = Counter([label for _, label in dataset])
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}
    weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float)
    return weights_tensor



# Hyperparameters
hyperparameters = {
    "batch_size": 128,
    "target_width": 300,
    "target_height": 200,
    "learning_rate": 0.001,
    "num_epochs": 150,
    "early_stopping_patience": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.2
    },
    "random_rotation_degrees": 20,
    "augment_prob" : 0.7
}

def ensure_width_gt_height(img):
    w, h = img.size
    if h > w:
        return img.transpose(PIL.Image.TRANSPOSE)
    else:
        return img

def resize_and_pad(img, target_width, target_height):
    # Calculate the aspect ratio
    img = ensure_width_gt_height(img)
    aspect_ratio = img.width / img.height
    target_aspect_ratio = target_width / target_height
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target, scale based on width
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than target, scale based on height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
    # Create a new image with the target size and paste the resized image
    new_img = PIL.Image.new("RGB", (target_width, target_height), (0, 0, 0))

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


class CustomTransform:

    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, img):
        return resize_and_pad(img, self.target_width, self.target_height)
    
    
# Model Architecture
class birdClassifier(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(birdClassifier, self).__init__()
        print("Initializing birdClassifier for 300x200 images")
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 10)  # Output layer for 10 bird classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupStepScheduler(_LRScheduler):

    """
    Custom learning rate scheduler with warmup period followed by step decay.
    Args:
        optimizer: The optimizer to modify
        init_lr: Initial learning rate during warmup (default: 0.001)
        max_lr: Maximum learning rate after warmup (default: 0.05)
        warmup_epochs: Number of epochs for warmup phase (default: 5)
        decay_epochs: Number of epochs between learning rate decay (default: 10)
        decay_factor: Factor to multiply learning rate by at each decay step (default: 0.5)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, init_lr=0.001, max_lr=0.05, warmup_epochs=5, 
                 decay_epochs=10, decay_factor=0.25, last_epoch=-1):

        self.init_lr = init_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        super(WarmupStepScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [self.init_lr + (self.max_lr - self.init_lr) * warmup_factor for _ in self.base_lrs]
            
        else:
            # Step decay every decay_epochs
            decay_power = math.floor((self.last_epoch - self.warmup_epochs) / self.decay_epochs)
            return [self.max_lr * (self.decay_factor ** decay_power) for _ in self.base_lrs]
        
    
    
# Now, create probabilistic augmentation for the training dataset
class ProbabilisticAugmentedTrainDataset():
    def __init__(self, dataset, augment_prob=0.7):
        self.dataset = dataset
        self.augment_prob = augment_prob
        self.images = []
        self.labels = []
        
        # Generate augmented versions and store with a probability
        for img, label in self.dataset:
            # Add the original image
            self.images.append(img)
            self.labels.append(label)
            
            # Add horizontal flip with a probability
            if random.random() < self.augment_prob:
                hflip_img = transforms.functional.hflip(img)
                self.images.append(hflip_img)
                self.labels.append(label)
            
            # Add vertical flip with a probability
            if random.random() < self.augment_prob:
                vflip_img = transforms.functional.vflip(img)
                self.images.append(vflip_img)
                self.labels.append(label)

            # Add random rotation with a probability
            if random.random() < self.augment_prob:
                rotation_angle = random.choice([-30, -15, 15, 30])  # Randomly select rotation angles
                rotated_img = transforms.functional.rotate(img, rotation_angle)
                self.images.append(rotated_img)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    
# Load the dataset
main_transform = transforms.Compose([
    CustomTransform(target_width=hyperparameters["target_width"], target_height=hyperparameters["target_height"]),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_path, transform=main_transform)



if mode == "train":
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        random_state=42,
        stratify=dataset.targets
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Apply probabilistic augmentation to the training dataset
    train_dataset = ProbabilisticAugmentedTrainDataset(train_dataset, augment_prob=hyperparameters["augment_prob"])

    # DataLoaders for train and validation
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
    
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset).to(device)

    # Initialize model, criterion, optimizer, and scheduler
    model = birdClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use CrossEntropyLoss with class weights
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    #optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=1e-4)
    scheduler = WarmupStepScheduler(optimizer, init_lr=0.001, max_lr=0.01, warmup_epochs=5, decay_epochs=10, decay_factor=0.5)

    # Training function
    def train_model():
        print("Starting training...")
        
        start_time = time.time()
        
        best_val_acc = 0
        best_model_state = None

        for epoch in range(hyperparameters["num_epochs"]):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            if ((time.time()-start_time)>6500):         # safety breaking
                break

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

            train_accuracy = correct_predictions.double() / total_predictions
            print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy:.4f}")

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            correct_val_preds = 0
            val_total_predictions = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_val_preds += torch.sum(preds == labels)
                    val_total_predictions += labels.size(0)

            val_accuracy = correct_val_preds.double() / val_total_predictions
            print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, model_path)
                print(f"New best model saved with accuracy: {best_val_acc:.4f}")

    # Train the model
    train_model()
    
elif mode == "test":
    model = birdClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())

    # Save predictions to CSV
    with open('bird.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        writer.writerows([[label] for label in predictions])

    print("Predictions saved to bird.csv")

print("Process completed.")
