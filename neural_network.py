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
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
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
            x: input vector (flattened)
        
        Returns:
            output: predictions from network
        """
        self.activations = [x.reshape(-1, 1)]
        self.pre_activations = []
        
        a = x.reshape(-1, 1)
        
        for i in range(len(self.weights)):
            z = self.weights[i].dot(a) + self.biases[i]
            self.pre_activations.append(z)
            
            is_last = (i == len(self.weights) - 1)
            a = self.get_activation(z, is_last=is_last)
            self.activations.append(a)
        
        return a
    
    def softmax(self, z):
        """Softmax for multi-class classification"""
        z = z - np.max(z)
        exp = np.exp(z)
        return exp / exp.sum()
    
    def cross_entropy_loss(self, predictions, target):
        """
        Cross-entropy loss for classification.
        
        Args:
            predictions: network output (logits or probabilities)
            target: one-hot encoded target
        
        Returns:
            loss: scalar loss value
        """
        # Apply softmax to get probabilities
        probs = self.softmax(predictions.ravel())
        
        # Avoid log(0)
        epsilon = 1e-7
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Cross-entropy: -sum(target * log(probs))
        loss = -np.sum(target * np.log(probs))
        return loss
    
    def backward_pass(self, target):
        """
        Backpropagation: compute deltas and gradients.
        
        Args:
            target: one-hot encoded target
        
        Returns:
            dW_list, db_list: gradients for each layer
        """
        dW_list = []
        db_list = []
        
        # Output layer delta
        # δ^L = (a^L - y) ⊙ σ'(z^L)
        # For softmax + cross-entropy, δ^L = a^L - y
        probs = self.softmax(self.pre_activations[-1].ravel())
        delta = (probs - target).reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Weight gradient: dW = δ * (a^(l-1))^T
            dW = delta.dot(self.activations[i].T)
            
            # Bias gradient: db = δ
            db = delta
            
            dW_list.insert(0, dW)
            db_list.insert(0, db)
            
            # Propagate delta to previous layer
            # δ^(l-1) = (W^(l))^T * δ^l ⊙ σ'(z^(l-1))
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
        Train for one epoch.
        
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
            
            batch_loss = 0
            
            for idx in batch_indices:
                x = X_train[idx]
                y = y_train[idx]
                
                # Forward pass
                output = self.forward_pass(x)
                
                # Compute loss
                loss = self.cross_entropy_loss(output, y)
                batch_loss += loss
                
                # Backward pass
                dW_list, db_list = self.backward_pass(y)
                
                # Update weights
                self.update_weights(dW_list, db_list)
            
            epoch_loss += batch_loss
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate accuracy on test set.
        
        Args:
            X_test: test data
            y_test: test labels (one-hot encoded)
        
        Returns:
            accuracy: percentage of correct predictions
            avg_loss: average loss
        """
        correct = 0
        total_loss = 0
        
        for i in range(X_test.shape[0]):
            x = X_test[i]
            y = y_test[i]
            
            output = self.forward_pass(x)
            _, pred = self.predict(x)
            true_label = np.argmax(y)
            
            if pred == true_label:
                correct += 1
            
            loss = self.cross_entropy_loss(output, y)
            total_loss += loss
        
        accuracy = (correct / X_test.shape[0]) * 100
        avg_loss = total_loss / X_test.shape[0]
        
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
    
    print(f"Training Neural Network")
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
