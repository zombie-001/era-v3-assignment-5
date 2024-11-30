import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import SimpleCNN
import glob
import pytest

def get_latest_model():
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has too many parameters: {total_params}"

def test_model_accuracy():
    # Load the model
    model = SimpleCNN()
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is too low: {accuracy:.2f}%"
    print(f"Model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    pytest.main([__file__]) 