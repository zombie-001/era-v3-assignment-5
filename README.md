# ML Model CI/CD Pipeline

This repository implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions.

## Project Overview

- **Task**: MNIST Handwritten Digit Classification (10 classes)
- **Input**: 28x28 grayscale images
- **Framework**: PyTorch
- **Training**: Single epoch training with batch size 128
- **Target**: >80% accuracy with <25,000 parameters

## Model Architecture

```
Input Image: 1x28x28 (Grayscale)
│
├── Conv1: 1→8 channels, 3x3 kernel
│   ├── BatchNorm2d
│   └── ReLU
│   └── Output: 8x26x26
│
├── Conv2: 8→16 channels, 3x3 kernel
│   ├── BatchNorm2d
│   └── ReLU
│   └── Output: 16x24x24
│
├── MaxPool2d: 2x2, stride 2
│   └── Output: 16x12x12
│
├── Conv3: 16→8 channels, 1x1 kernel (dimensionality reduction)
│   ├── BatchNorm2d
│   └── ReLU
│   └── Output: 8x12x12
│
├── Conv4: 8→16 channels, 3x3 kernel
│   ├── BatchNorm2d
│   └── ReLU
│   └── Output: 16x10x10
│
├── Conv5: 16→8 channels, 3x3 kernel
│   ├── BatchNorm2d
│   └── ReLU
│   └── Output: 8x8x8
│
└── Fully Connected: 512→10 (8*8*8 → 10)
```

## Model Parameters
- Conv1: 80 parameters (1×8×3×3 + 8)
- Conv2: 1,168 parameters (8×16×3×3 + 16)
- Conv3: 136 parameters (16×8×1×1 + 8)
- Conv4: 1,168 parameters (8×16×3×3 + 16)
- Conv5: 1,160 parameters (16×8×3×3 + 8)
- FC: 5,130 parameters (8×8×8×10 + 10)
- BatchNorm parameters: ~96 parameters
- Total Parameters: ~8,938

## Features
- BatchNormalization after each convolution for better training
- ReLU activation functions for non-linearity
- MaxPooling for spatial dimension reduction
- 1x1 convolution for channel dimension reduction
- Single fully connected layer for final classification
- Adam optimizer with learning rate 0.01
- Cross Entropy Loss function

## CI/CD Pipeline
- Automated model training
- Parameter count verification (<25,000)
- Input/output shape validation (28x28 input, 10 outputs)
- Accuracy testing (>80% requirement)
- Model artifact storage with timestamp and accuracy

## Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python model.py
```

4. Run tests:
```bash
pytest test_model.py -v
```

## GitHub Actions Workflow
The pipeline automatically:
1. Sets up Python 3.8 environment
2. Installs CPU version of PyTorch and dependencies
3. Trains the model for one epoch
4. Runs validation tests:
   - Checks model architecture
   - Verifies parameter count
   - Tests model accuracy
5. Stores the trained model as an artifact (90 days retention)

## Model Performance Monitoring
During training, you can monitor:
- Real-time loss values
- Training accuracy per batch
- Final training accuracy
- Test set accuracy
- Total parameter count
- Detailed layer-wise parameter breakdown

## Requirements
- Python 3.8+
- PyTorch 2.2.0 (CPU version)
- torchvision 0.17.0 (CPU version)
- pytest ≥6.0.0
- numpy ≥1.19.0
- tqdm ≥4.65.0

## Notes
- Model is optimized for CPU training
- Uses CPU-only version of PyTorch for CI/CD compatibility
- Includes SSL certificate handling for dataset download
- Model files are saved with timestamp and accuracy for versioning
- All tests must pass before deployment
- Training progress is displayed with tqdm progress bars