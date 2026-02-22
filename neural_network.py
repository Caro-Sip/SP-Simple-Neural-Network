import os
# Enable multi-threading for NumPy operations
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('MKL_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count()))

import numpy as np
from collections import defaultdict


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, activation='sigmoid'):
        """
        Initialize neural network with random weights and biases.
        
        Args:
            layer_sizes: list of integers representing neurons in each layer
            learning_rate: η for gradient descent
            activation: activation function ('sigmoid', 'relu', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for sigmoid/tanh
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])  # He initialization
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])  # Xavier initialization
            
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(W)
            self.biases.append(b)
        
        # Store activations for backprop
        self.activations = []
        self.pre_activations = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def get_activation(self, z, is_last=False):
        """Apply activation function (linear for last layer)"""
        if is_last:
            return z  # linear activation for output
        
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
    
    def get_activation_derivative(self, z, is_last=False):
        """Derivative of activation function"""
        if is_last:
            return np.ones_like(z)  # derivative of linear is 1
        
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation == 'relu':
            return self.relu_derivative(z)
        elif self.activation == 'tanh':
            return self.tanh_derivative(z)
    
    def forward_pass(self, x):
        """
        Forward pass through network.
        
        Args:
            x: input vector (flattened) or batch (batch_size, features)
        
        Returns:
            output: predictions from network
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # (features, 1)
        else:
            x = x.T  # (features, batch_size)
        
        self.activations = [x]
        self.pre_activations = []
        
        a = x
        
        for i in range(len(self.weights)):
            z = self.weights[i].dot(a) + self.biases[i]
            self.pre_activations.append(z)
            
            is_last = (i == len(self.weights) - 1)
            a = self.get_activation(z, is_last=is_last)
            self.activations.append(a)
        
        return a
    
    def softmax(self, z):
        """Softmax for multi-class classification (handles batches)"""
        if z.ndim == 1:
            z = z - np.max(z)
            exp = np.exp(z)
            return exp / exp.sum()
        else:
            # Batch mode: z is (classes, batch_size)
            z = z - np.max(z, axis=0, keepdims=True)
            exp = np.exp(z)
            return exp / exp.sum(axis=0, keepdims=True)
    
    def cross_entropy_loss(self, predictions, target):
        """
        Cross-entropy loss for classification.
        
        Args:
            predictions: network output (logits) - (classes, 1) or (classes, batch_size)
            target: one-hot encoded target - (classes,) or (batch_size, classes)
        
        Returns:
            loss: scalar loss value (averaged over batch)
        """
        # Handle single sample
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            probs = self.softmax(predictions.ravel())
            epsilon = 1e-7
            probs = np.clip(probs, epsilon, 1 - epsilon)
            loss = -np.sum(target * np.log(probs))
            return loss
        
        # Batch mode: predictions is (classes, batch_size), target is (batch_size, classes)
        probs = self.softmax(predictions)  # (classes, batch_size)
        epsilon = 1e-7
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Cross-entropy: -sum(target * log(probs)) averaged over batch
        # target.T is (classes, batch_size)
        loss = -np.sum(target.T * np.log(probs)) / predictions.shape[1]
        return loss
    
    def backward_pass(self, target):
        """
        Backpropagation: compute deltas and gradients.
        
        Args:
            target: one-hot encoded target - (classes,) or (batch_size, classes)
        
        Returns:
            dW_list, db_list: gradients for each layer (averaged over batch)
        """
        dW_list = []
        db_list = []
        
        # Determine batch size
        if self.pre_activations[-1].shape[1] == 1:
            # Single sample
            batch_size = 1
            probs = self.softmax(self.pre_activations[-1].ravel())
            delta = (probs - target).reshape(-1, 1)
        else:
            # Batch mode
            batch_size = self.pre_activations[-1].shape[1]
            probs = self.softmax(self.pre_activations[-1])  # (classes, batch_size)
            # target is (batch_size, classes), need (classes, batch_size)
            delta = probs - target.T  # (classes, batch_size)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Weight gradient: dW = δ * (a^(l-1))^T / batch_size
            dW = delta.dot(self.activations[i].T) / batch_size
            
            # Bias gradient: db = mean(δ) over batch
            db = np.mean(delta, axis=1, keepdims=True)
            
            dW_list.insert(0, dW)
            db_list.insert(0, db)
            
            # Propagate delta to previous layer
            if i > 0:
                delta = self.weights[i].T.dot(delta) * self.get_activation_derivative(
                    self.pre_activations[i-1], is_last=False
                )
        
        return dW_list, db_list
    
    def update_weights(self, dW_list, db_list):
        """
        Update weights and biases using gradient descent.
        
        Args:
            dW_list: weight gradients
            db_list: bias gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW_list[i]
            self.biases[i] -= self.learning_rate * db_list[i]
    
    def predict(self, x):
        """Get prediction for single sample"""
        output = self.forward_pass(x)
        probs = self.softmax(output.ravel())
        return probs, np.argmax(probs)
    
    def train_epoch(self, X_train, y_train, batch_size=32, shuffle=True):
        """
        Train for one epoch using vectorized batch processing.
        
        Args:
            X_train: training data (N, features)
            y_train: training labels (N, 10) one-hot encoded
            batch_size: batch size
            shuffle: whether to shuffle data
        
        Returns:
            epoch_loss: average loss over epoch
        """
        N = X_train.shape[0]
        indices = np.arange(N)
        
        if shuffle:
            np.random.shuffle(indices)
        
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_indices = indices[start_idx:end_idx]
            
            # Get entire batch at once
            X_batch = X_train[batch_indices]  # (batch_size, features)
            y_batch = y_train[batch_indices]  # (batch_size, classes)
            
            # Forward pass for entire batch
            output = self.forward_pass(X_batch)
            
            # Compute loss for entire batch
            loss = self.cross_entropy_loss(output, y_batch)
            epoch_loss += loss * len(batch_indices)  # Scale by batch size for averaging
            
            # Backward pass for entire batch (gradients already averaged)
            dW_list, db_list = self.backward_pass(y_batch)
            
            # Update weights once per batch
            self.update_weights(dW_list, db_list)
            
            num_batches += 1
        
        return epoch_loss / N
    
    def evaluate(self, X_test, y_test, batch_size=512):
        """
        Evaluate accuracy on test set using vectorized batches.
        
        Args:
            X_test: test data
            y_test: test labels (one-hot encoded)
            batch_size: batch size for evaluation (larger is faster)
        
        Returns:
            accuracy: percentage of correct predictions
            avg_loss: average loss
        """
        N = X_test.shape[0]
        total_correct = 0
        total_loss = 0
        
        # Process in batches for speed
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            X_batch = X_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            # Forward pass for batch
            output = self.forward_pass(X_batch)  # (classes, batch_size)
            
            # Get predictions: argmax along class dimension
            probs = self.softmax(output)  # (classes, batch_size)
            predictions = np.argmax(probs, axis=0)  # (batch_size,)
            true_labels = np.argmax(y_batch, axis=1)  # (batch_size,)
            
            # Count correct predictions
            total_correct += np.sum(predictions == true_labels)
            
            # Compute loss
            loss = self.cross_entropy_loss(output, y_batch)
            total_loss += loss * len(X_batch)
        
        accuracy = (total_correct / N) * 100
        avg_loss = total_loss / N
        
        return accuracy, avg_loss


def train_network(X_train, y_train, X_val, y_val, epochs=10, batch_size=32, 
                  layer_sizes=[784, 128, 64, 10], learning_rate=0.01, activation='sigmoid'):
    """
    Full training loop with validation and logging.
    
    Args:
        X_train, y_train: training data and labels
        X_val, y_val: validation data and labels
        epochs: number of epochs
        batch_size: batch size
        layer_sizes: architecture
        learning_rate: learning rate η
        activation: activation function ('sigmoid', 'relu', 'tanh')
    
    Returns:
        model: trained NeuralNetwork
        history: dict with loss/accuracy per epoch
    """
    model = NeuralNetwork(layer_sizes, learning_rate=learning_rate, activation=activation)
    
    history = defaultdict(list)
    
    print("Training Neural Network")
    print(f"Architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Activation: {activation}, Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        train_loss = model.train_epoch(X_train, y_train, batch_size=batch_size)
        
        # Validation
        val_acc, val_loss = model.evaluate(X_val, y_val)
        train_acc, _ = model.evaluate(X_train, y_train)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    
    print("=" * 60)
    print("Training Complete!")
    
    return model, history
