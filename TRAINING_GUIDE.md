# Neural Network Training Guide

## Overview
This project now includes a complete neural network implementation with backpropagation for training on the MNIST dataset.

## New Files Created

### 1. `neural_network.py`
Contains the `NeuralNetwork` class with:
- Forward propagation
- Backpropagation (gradient computation)
- Weight updates (gradient descent)
- Training loop (train_epoch)
- Evaluation metrics (accuracy, loss)
- Multiple activation functions (sigmoid, relu, tanh)

### 2. `train_mnist.py`
Complete training script that:
- Loads MNIST data from ubyte files
- Normalizes images to [0, 1]
- Converts labels to one-hot encoding
- Splits data into train/validation/test sets
- Trains the neural network
- Evaluates performance
- Visualizes results with plots

## How to Run

### Quick Start
Simply run the training script:
```bash
python train_mnist.py
```

### What It Does
1. Loads 60,000 training images and 10,000 test images from `mnist_data/`
2. Splits training data into 80% train (48,000) and 20% validation (12,000)
3. Trains a neural network with architecture: 784 → 128 → 64 → 10
4. Prints progress every 5 epochs
5. Evaluates on test set
6. Shows training curves and sample predictions

### Expected Output
```
MNIST Neural Network Training
============================================================

1. Loading MNIST dataset...
Loading 60000 images (28x28)...
Loading 60000 labels...
Loading 10000 images (28x28)...
Loading 10000 labels...

2. Converting labels to one-hot encoding...
3. Splitting data into train/validation sets...
Training samples: 48000
Validation samples: 12000

4. Training neural network...
Training Neural Network
Architecture: 784 -> 128 -> 64 -> 10
Activation: sigmoid, Learning Rate: 0.01
Epochs: 50, Batch Size: 32
============================================================
Epoch   1/50
  Train Loss: 2.3456, Train Acc: 12.34%
  Val Loss:   2.3012, Val Acc:   15.67%
...

5. Evaluating on test set...
Test Accuracy: 92.45%
Test Loss: 0.2456
```

## Customizing Training

You can modify the hyperparameters in `train_mnist.py`:

```python
# Network architecture (input, hidden layers..., output)
layer_sizes = [784, 256, 128, 10]  # Larger network

# Learning rate
learning_rate = 0.05  # Faster learning

# Number of training epochs
epochs = 100  # More training

# Batch size
batch_size = 64  # Larger batches

# Activation function
activation = 'relu'  # Options: 'sigmoid', 'relu', 'tanh'
```

## Using the Trained Model

After training, you can use the model for predictions:

```python
from neural_network import NeuralNetwork
import numpy as np

# Load a trained model (or train a new one)
model = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
# ... train the model ...

# Make a prediction on a single image
image = X_test[0]  # Shape: (784,)
probs, prediction = model.predict(image)

print(f"Predicted digit: {prediction}")
print(f"Confidence: {probs[prediction]:.4f}")
print(f"All probabilities: {probs}")
```

## Integration with Existing Code

The new neural network can work alongside your existing code:

- `main.py` - Your original feedforward demo (unchanged)
- `ExampleXORnetwork.py` - XOR training example (unchanged)
- `dataset60Reader.py` - MNIST reader functions (unchanged)
- `neural_network.py` - **NEW**: Complete NN class with training
- `train_mnist.py` - **NEW**: MNIST training script

## Performance Tips

1. **For faster training**: Use `activation='relu'` and increase learning rate to 0.05
2. **For better accuracy**: Increase epochs to 100 or add more hidden neurons
3. **For memory efficiency**: Reduce batch_size if running out of memory
4. **For debugging**: Print loss every epoch by changing `if (epoch + 1) % 5 == 0` to `if True`

## Project Structure

```
SP-Simple-Neural-Network/
│
├── neural_network.py       # ← NEW: Neural network class
├── train_mnist.py          # ← NEW: Training script
│
├── main.py                 # Original feedforward demo
├── ExampleXORnetwork.py    # XOR training example
├── dataset60Reader.py      # MNIST utilities
├── readubyte.py           # Basic MNIST reader
├── Download_mnistdatasets.py
│
└── mnist_data/             # MNIST dataset
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## Next Steps

1. Run `python train_mnist.py` to train your first model
2. Experiment with different architectures and hyperparameters
3. Try different activation functions (relu, tanh)
4. Implement additional features:
   - Save/load trained models
   - Add learning rate decay
   - Implement momentum or Adam optimizer
   - Add dropout for regularization
   - Use mini-batch gradient descent averaging

## Troubleshooting

**"No module named matplotlib"**
```bash
pip install matplotlib
```

**"Training is very slow"**
- Reduce number of training samples
- Increase batch_size
- Reduce number of epochs
- Try activation='relu' instead of 'sigmoid'

**"Accuracy is low"**
- Train for more epochs (50-100)
- Try different activation functions
- Adjust learning rate (try 0.001 to 0.1)
- Check if data is normalized correctly
