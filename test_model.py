import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import SimpleCNN
import glob
import pytest
import numpy as np

def get_latest_model():
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def print_test_header(test_name):
    print(f"\n{'='*50}")
    print(f"Running Test: {test_name}")
    print('='*50)

def test_model_architecture():
    print_test_header("Model Architecture Test")
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print("✓ Input shape (1, 1, 28, 28) -> Output shape", output.shape)
    assert output.shape == (1, 10), "Model output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    assert total_params < 25000, f"Model has too many parameters: {total_params}"

def test_batch_processing():
    print_test_header("Batch Processing Test")
    model = SimpleCNN()
    batch_sizes = [1, 4, 16, 32, 64]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        print(f"✓ Successfully processed batch size: {batch_size}")
        assert output.shape == (batch_size, 10), f"Failed batch processing for size {batch_size}"

def test_model_components():
    print_test_header("Model Components Test")
    model = SimpleCNN()
    
    components = {
        'Convolution Layers': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
        'Batch Normalization': ['bn1', 'bn2', 'bn3', 'bn4', 'bn5'],
        'Other Layers': ['fc']
    }
    
    for category, layers in components.items():
        print(f"\nChecking {category}:")
        for layer in layers:
            print(f"✓ Found {layer}")
            assert hasattr(model, layer), f"Missing {layer} layer"

def test_model_gradients():
    print_test_header("Gradient Flow Test")
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(test_input)
    loss = output.sum()
    loss.backward()
    print("✓ Gradients computed successfully")
    assert test_input.grad is not None, "Model is not computing gradients properly"

def test_model_accuracy():
    print_test_header("Model Accuracy Test")
    model = SimpleCNN()
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            for i in range(target.size(0)):
                label = target[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    print(f"\n✓ Overall Model Accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy is too low: {accuracy:.2f}%"
    
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"✓ Digit {i}: {class_acc:.2f}%")
        assert class_acc > 70, f"Accuracy for digit {i} is too low: {class_acc:.2f}%"

def test_model_robustness():
    print_test_header("Model Robustness Test")
    model = SimpleCNN()
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_input = torch.randn(1, 1, 28, 28)
    noise_levels = [0.1, 0.2, 0.3]
    
    with torch.no_grad():
        for noise in noise_levels:
            noisy_input = test_input + noise * torch.randn_like(test_input)
            output = model(noisy_input)
            print(f"✓ Model handled noise level: {noise:.1f}")
            assert output.shape == (1, 10), f"Model failed with noise level {noise}"

if __name__ == "__main__":
    pytest.main([__file__]) 