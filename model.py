import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import ssl
import os
from tqdm import tqdm

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Reduced number of filters and added batch normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Reduced from 32 to 16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)  # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3
        x = x.view(-1, 32 * 3 * 3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def print_model_summary(model):
    print("\n" + "="*50)
    print("Model Architecture Summary")
    print("="*50)
    
    print("\nLayer Details:")
    print("-"*50)
    total_params = 0
    
    # Conv1 layer
    conv1_params = sum(p.numel() for p in model.conv1.parameters())
    print(f"Conv1 Layer: 1 -> 16 channels, 3x3 kernel")
    print(f"Output shape: 28x28 -> 14x14 (after pooling)")
    print(f"Parameters: {conv1_params:,}")
    total_params += conv1_params
    
    # Conv2 layer
    conv2_params = sum(p.numel() for p in model.conv2.parameters())
    print(f"\nConv2 Layer: 16 -> 32 channels, 3x3 kernel")
    print(f"Output shape: 14x14 -> 7x7 (after pooling)")
    print(f"Parameters: {conv2_params:,}")
    total_params += conv2_params
    
    # Conv3 layer
    conv3_params = sum(p.numel() for p in model.conv3.parameters())
    print(f"\nConv3 Layer: 32 -> 32 channels, 3x3 kernel")
    print(f"Output shape: 7x7 -> 3x3 (after pooling)")
    print(f"Parameters: {conv3_params:,}")
    total_params += conv3_params
    
    # FC1 layer
    fc1_params = sum(p.numel() for p in model.fc1.parameters())
    print(f"\nFC1 Layer: {32 * 3 * 3} -> 64 neurons")
    print(f"Parameters: {fc1_params:,}")
    total_params += fc1_params
    
    # FC2 layer
    fc2_params = sum(p.numel() for p in model.fc2.parameters())
    print(f"\nFC2 Layer: 64 -> 10 neurons (output)")
    print(f"Parameters: {fc2_params:,}")
    total_params += fc2_params
    
    # BatchNorm parameters
    bn_params = sum(p.numel() for p in model.bn1.parameters()) + \
                sum(p.numel() for p in model.bn2.parameters()) + \
                sum(p.numel() for p in model.bn3.parameters())
    print(f"\nBatch Normalization Layers")
    print(f"Parameters: {bn_params:,}")
    total_params += bn_params
    
    print("\n" + "="*50)
    print(f"Total Parameters: {total_params:,}")
    print("="*50 + "\n")

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train_model():
    print("Starting training process...")
    
    # Initialize model and print summary
    model = SimpleCNN()
    print_model_summary(model)
    
    # Load MNIST dataset with augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        print("Downloading and loading MNIST dataset...")
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
        print(f"Dataset loaded successfully! Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    print("\nStarting training...")
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pbar = tqdm(train_loader, desc='Training', 
                unit='batch', 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
        
        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            train_acc = 100 * correct_train / total_train
            pbar.set_description(f'Training (loss={avg_loss:.4f}, acc={train_acc:.2f}%)')
    
    # Calculate final training accuracy
    final_train_acc = 100 * correct_train / total_train
    print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
    
    # Calculate validation accuracy
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'mnist_model_{timestamp}_acc{test_accuracy:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Print final metrics
    print("\n" + "="*50)
    print("Final Training Metrics:")
    print(f"Training Accuracy: {final_train_acc:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Loss: {avg_loss:.4f}")
    print("="*50)
    
    return model

if __name__ == "__main__":
    train_model() 