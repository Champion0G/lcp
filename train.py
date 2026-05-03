import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import *
from resnet_model import LungCancerResNet
from utils.visualization import plot_training_history

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_data_loaders():
    """
    Sets up the dataset and returns data loaders for training and validation.
    """
    print(f"Loading dataset from: {DATASET_DIR}")
    
    # Check if dataset directory is empty
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"Warning: Dataset directory {DATASET_DIR} is empty or does not exist.")
        print("Please place 'benign' and 'malignant' folders inside it.")
        return None, None
    
    # 1. Define Transforms
    # Training transforms include augmentation to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms only include resizing and normalization
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load the full dataset (initially without transforms)
    # We will apply transforms after splitting via custom Dataset wrapper if we wanted perfectly strict separate transforms,
    # but for simplicity and standard PyTorch ImageFolder practice, we can apply train transforms to all, 
    # OR we can load twice and split based on indices. 
    # A cleaner approach for standard PyTorch datasets when needing different transforms for train/val:
    
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    
    # 3. Train-Validation Split (80/20)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_subset, transform=val_transforms)
    
    print(f"Total images: {dataset_size}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    return train_loader, val_loader

def train_model():
    """
    Main training pipeline.
    """
    train_loader, val_loader = get_data_loaders()
    if train_loader is None or val_loader is None:
        return
        
    print(f"Using device: {DEVICE}")
    
    # Initialize Model
    model = LungCancerResNet(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    
    # Loss Function and Optimizer
    # Optional: Calculate class weights if dataset is imbalanced
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Tracking history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"- Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} "
              f"- Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
              
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model to {MODEL_SAVE_PATH}")
            
    print("Training complete.")
    
    # Plot history
    plot_training_history(history['train_loss'], history['val_loss'], 
                          history['train_acc'], history['val_acc'], OUTPUTS_DIR)
    print(f"Saved training history plot to {OUTPUTS_DIR}")

if __name__ == "__main__":
    train_model()
