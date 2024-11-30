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
        # First two conv layers without pooling
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)     # 28x28 -> 26x26
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)    # 26x26 -> 24x24
        self.bn2 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2, 2)                            # 24x24 -> 12x12
        
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1)             # 12x12 -> 12x12 (1x1 conv)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1)    # 12x12 -> 10x10
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1)    # 10x10 -> 8x8
        self.bn5 = nn.BatchNorm2d(8)
        
        self.fc = nn.Linear(8 * 8 * 8, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))      # 28x28 -> 26x26
        x = self.relu(self.bn2(self.conv2(x)))      # 26x26 -> 24x24
        x = self.pool(x)                            # 24x24 -> 12x12
        x = self.relu(self.bn3(self.conv3(x)))      # 12x12 -> 12x12
        x = self.relu(self.bn4(self.conv4(x)))      # 12x12 -> 10x10
        x = self.relu(self.bn5(self.conv5(x)))      # 10x10 -> 8x8
        x = x.view(-1, 8 * 8 * 8)                   # Flatten
        x = self.fc(x)                              # Output
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
    print(f"Conv1 Layer: 1 -> 8 channels, 3x3 kernel")
    print(f"Output shape: 28x28 -> 26x26")
    print(f"Parameters: {conv1_params:,}")
    total_params += conv1_params
    
    # Conv2 layer
    conv2_params = sum(p.numel() for p in model.conv2.parameters())
    print(f"\nConv2 Layer: 8 -> 16 channels, 3x3 kernel")
    print(f"Output shape: 26x26 -> 24x24")
    print(f"Parameters: {conv2_params:,}")
    total_params += conv2_params
    
    # MaxPool layer
    print(f"\nMaxPool Layer: 2x2, stride 2")
    print(f"Output shape: 24x24 -> 12x12")
    print(f"Parameters: 0")
    
    # Conv3 layer (1x1)
    conv3_params = sum(p.numel() for p in model.conv3.parameters())
    print(f"\nConv3 Layer: 16 -> 8 channels, 1x1 kernel")
    print(f"Output shape: 12x12 -> 12x12")
    print(f"Parameters: {conv3_params:,}")
    total_params += conv3_params
    
    # Conv4 layer
    conv4_params = sum(p.numel() for p in model.conv4.parameters())
    print(f"\nConv4 Layer: 8 -> 16 channels, 3x3 kernel")
    print(f"Output shape: 12x12 -> 10x10")
    print(f"Parameters: {conv4_params:,}")
    total_params += conv4_params
    
    # Conv5 layer
    conv5_params = sum(p.numel() for p in model.conv5.parameters())
    print(f"\nConv5 Layer: 16 -> 8 channels, 3x3 kernel")
    print(f"Output shape: 10x10 -> 8x8")
    print(f"Parameters: {conv5_params:,}")
    total_params += conv5_params
    
    # FC layer
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print(f"\nFC Layer: {8 * 8 * 8} -> 10 neurons (output)")
    print(f"Parameters: {fc_params:,}")
    total_params += fc_params
    
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
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        print("Downloading and loading MNIST dataset...")
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        print(f"Dataset loaded successfully! Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
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