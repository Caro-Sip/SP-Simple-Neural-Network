import os
# CRITICAL: Set thread count BEFORE importing NumPy
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

import struct
import numpy as np
import matplotlib.pyplot as plt
from neural_network import train_network


def load_mnist_images(image_path):
    """
    Load all MNIST images from idx3-ubyte file.
    
    Args:
        image_path: Path to images idx3-ubyte file
    
    Returns:
        images: numpy array of shape (N, 784) normalized to [0, 1]
    """
    with open(image_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"Loading {num_images} images ({rows}x{cols})...")
        
        # Read all image data
        buf = f.read(rows * cols * num_images)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        
        # Reshape to (num_images, 784) and normalize to [0, 1]
        images = images.reshape(num_images, rows * cols) / 255.0
        
    return images


def load_mnist_labels(label_path):
    """
    Load all MNIST labels from idx1-ubyte file.
    
    Args:
        label_path: Path to labels idx1-ubyte file
    
    Returns:
        labels: numpy array of shape (N,) with integer labels 0-9
    """
    with open(label_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        print(f"Loading {num_labels} labels...")
        
        # Read all label data
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        
    return labels


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: array of integer labels (N,)
        num_classes: number of classes (default: 10 for MNIST)
    
    Returns:
        one_hot: array of shape (N, num_classes)
    """
    N = labels.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels] = 1
    return one_hot


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, X_test, y_test, num_samples=10):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: trained NeuralNetwork
        X_test: test images
        y_test: test labels (one-hot encoded)
        num_samples: number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Randomly select samples
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        x = X_test[idx]
        y_true = np.argmax(y_test[idx])
        
        # Get prediction
        probs, y_pred = model.predict(x)
        
        # Reshape image to 28x28 for display
        image = x.reshape(28, 28)
        
        # Display image
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        
        # Color: green if correct, red if incorrect
        color = 'green' if y_pred == y_true else 'red'
        axes[i].set_title(f'True: {y_true}, Pred: {y_pred}', color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("MNIST Neural Network Training")
    print("=" * 60)
    
    # ========== Load MNIST Data ==========
    print("\n1. Loading MNIST dataset...")
    X_train = load_mnist_images('mnist_data/train-images-idx3-ubyte')
    y_train = load_mnist_labels('mnist_data/train-labels-idx1-ubyte')
    X_test = load_mnist_images('mnist_data/t10k-images-idx3-ubyte')
    y_test = load_mnist_labels('mnist_data/t10k-labels-idx1-ubyte')
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # ========== One-Hot Encode Labels ==========
    print("\n2. Converting labels to one-hot encoding...")
    y_train_encoded = one_hot_encode(y_train, num_classes=10)
    y_test_encoded = one_hot_encode(y_test, num_classes=10)
    
    # ========== Split Training into Train/Validation ==========
    print("\n3. Splitting data into train/validation sets...")
    val_split = int(0.2 * len(X_train))
    X_val = X_train[:val_split]
    y_val = y_train_encoded[:val_split]
    X_train_split = X_train[val_split:]
    y_train_split = y_train_encoded[val_split:]
    
    print(f"Training samples: {X_train_split.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # ========== Train Neural Network ==========
    print("\n4. Training neural network...")
    
    # Network hyperparameters
    layer_sizes = [784, 256, 128, 10]  # Increased hidden layers for more computation
    learning_rate = 0.1  # Increased for larger batch size (0.01 * 256/32 ≈ 0.08)
    epochs = 20
    batch_size = 256  # Larger batches to saturate all CPU cores
    activation = 'relu'  # ReLU is faster and learns better than sigmoid
    
    model, history = train_network(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        activation=activation
    )
    
    # ========== Evaluate on Test Set ==========
    print("\n5. Evaluating on test set...")
    test_acc, test_loss = model.evaluate(X_test, y_test_encoded)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # ========== Visualize Results ==========
    print("\n6. Visualizing results...")
    
    try:
        # Plot training history
        plot_training_history(history)
        
        # Show some predictions
        visualize_predictions(model, X_test, y_test_encoded, num_samples=10)
        
    except Exception as e:
        print(f"Could not display plots: {e}")
        print("Training data saved in 'history' variable")
    
    # ========== Make Sample Predictions ==========
    print("\n7. Sample predictions:")
    print("-" * 60)
    for i in range(5):
        x = X_test[i]
        y_true = y_test[i]
        probs, y_pred = model.predict(x)
        print(f"Sample {i+1}: True={y_true}, Predicted={y_pred}, Confidence={probs[y_pred]:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete! Model ready for use.")
    print("=" * 60)
