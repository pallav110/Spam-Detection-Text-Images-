import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
import random
import shutil
from sklearn.model_selection import train_test_split

class BalancedSpamImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.classes = ['personal_image_ham', 'personal_image_spam']
        self.images = []
        self.labels = []
        self.class_counts = {0: 0, 1: 0}  # For storing class distribution
        
        missing_folders = []
        # Load images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                missing_folders.append(class_name)
                continue
                
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
                self.class_counts[class_idx] += 1
        
        # Print dataset statistics and warnings
        print(f"\n{phase.capitalize()} Dataset Statistics:")
        if missing_folders:
            print(f"WARNING: The following folders are missing: {', '.join(missing_folders)}")
            print("Please create these folders and add appropriate images.")
            print("The model needs both ham and spam images to train properly.")
            if len(self.images) == 0:
                raise ValueError("No valid images found in the dataset. Please add images to both ham and spam folders.")
        
        print(f"Ham images: {self.class_counts[0]}")
        print(f"Spam images: {self.class_counts[1]}")
        
        if self.class_counts[0] == 0 or self.class_counts[1] == 0:
            raise ValueError(
                "One or both classes have no images. The model needs both ham and spam images to train properly.\n"
                f"Ham images: {self.class_counts[0]}\n"
                f"Spam images: {self.class_counts[1]}\n"
                "Please add images to both classes before training."
            )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default image in case of error
            default_image = torch.zeros((3, 224, 224))
            return default_image, self.labels[idx]

class SpamImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SpamImageClassifier, self).__init__()
        # Use a pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-8]:  # Keep last few layers trainable
            param.requires_grad = False
            
        # Modify the final layers for better feature extraction
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout for better regularization
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Add batch normalization for better training stability
        self.model.avgpool = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        return self.model(x)

def create_data_transforms():
    """Create data transforms with augmentation for training."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def calculate_class_weights(dataset):
    """Calculate class weights for balanced sampling."""
    labels = torch.tensor(dataset.labels)
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    # Calculate sample weights
    sample_weights = class_weights[labels]
    return sample_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=30, device='cuda', early_stopping_patience=5):
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            
            # Print classification report for best model
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, 
                                     target_names=['Ham', 'Spam']))
            
            # Create and save confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('visualizations/confusion_matrix.png')
            plt.close()
            
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_history.png')
    plt.close()
    
    return best_model_state, history, best_val_acc

def predict_image(model, image_path, device='cuda'):
    """Predict if an image is spam or ham."""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = float(probabilities[0][prediction]) * 100
        
        return prediction, confidence
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, 0.0

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Load datasets
    data_dir = 'data/Images'
    train_dataset = BalancedSpamImageDataset(data_dir, transform=train_transform, phase='train')
    val_dataset = BalancedSpamImageDataset(data_dir, transform=val_transform, phase='val')
    
    # Calculate weights for the sampler based on the full training dataset
    weights = calculate_class_weights(train_dataset)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = SpamImageClassifier()
    model = model.to(device)
    
    # Calculate class weights for loss function
    total_samples = len(train_dataset.labels)
    spam_ratio = train_dataset.class_counts[1] / total_samples
    ham_ratio = train_dataset.class_counts[0] / total_samples
    class_weights = torch.tensor([1/ham_ratio, 1/spam_ratio]).to(device)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train model
    print("\nStarting training...")
    best_model_state, history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, device=device, early_stopping_patience=5
    )
    
    # Save best model
    torch.save({
        'model_state_dict': best_model_state,
        'history': history,
        'best_val_acc': best_val_acc,
        'class_names': ['ham', 'spam']
    }, 'models/spam_image_model.pt')
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to models/spam_image_model.pt")