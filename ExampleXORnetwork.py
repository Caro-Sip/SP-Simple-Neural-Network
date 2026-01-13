import math

def sigmoid(h):
    """Sigmoid activation function: σ(h) = 1 / (1 + e^(-h))"""
    return 1 / (1 + math.exp(-h))

def neuron(x1, x2, w1, w2, b):
    """
    Single neuron with 2 inputs.
    Computes: σ(x1*w1 + x2*w2 + b)
    """
    h = x1 * w1 + x2 * w2 + b
    return sigmoid(h)

def forward_pass(x1, x2, w11, w21, b1, w12, w22, b2, v1, v2, c):
    """
    Forward propagation through the network.
    Architecture: 2 inputs -> 2 hidden neurons -> 1 output
    
    Returns: (h1, h2, y_hat) where h1, h2 are hidden layer outputs
    """
    # Hidden layer
    h1 = neuron(x1, x2, w11, w21, b1)
    h2 = neuron(x1, x2, w12, w22, b2)
    # Output layer
    y_hat = neuron(h1, h2, v1, v2, c)
    return h1, h2, y_hat

def backward_pass(x1, x2, y, h1, h2, y_hat, v1, v2):
    """
    Backpropagation: compute gradients for all weights and biases.
    Uses chain rule to propagate error from output back to hidden layer.
    
    Returns: dictionary of gradients for all parameters
    """
    # Output layer error (delta): ∂Loss/∂y_hat * ∂y_hat/∂z_out
    delta_out = (y_hat - y) * y_hat * (1 - y_hat)
    
    # Output layer gradients
    dv1 = delta_out * h1
    dv2 = delta_out * h2
    dc = delta_out
    
    # Hidden layer errors (backpropagate through output weights)
    delta_h1 = delta_out * v1 * h1 * (1 - h1)
    delta_h2 = delta_out * v2 * h2 * (1 - h2)
    
    # Hidden layer gradients
    dw11 = delta_h1 * x1
    dw21 = delta_h1 * x2
    db1 = delta_h1
    
    dw12 = delta_h2 * x1
    dw22 = delta_h2 * x2
    db2 = delta_h2
    
    return {
        'dv1': dv1, 'dv2': dv2, 'dc': dc,
        'dw11': dw11, 'dw21': dw21, 'db1': db1,
        'dw12': dw12, 'dw22': dw22, 'db2': db2
    }

def update_weights(params, gradients, learning_rate):
    """
    Update all network parameters using gradient descent.
    θ_new = θ_old - η * ∂Loss/∂θ
    """
    for key in params:
        params[key] -= learning_rate * gradients['d' + key]
    return params

def compute_loss(dataset, params):
    """Calculate mean squared error across entire dataset."""
    total_loss = 0
    for x1, x2, y in dataset:
        _, _, y_hat = forward_pass(x1, x2, **params)
        total_loss += (y_hat - y) ** 2
    return total_loss / len(dataset)

def train_network(dataset, epochs=10000, learning_rate=0.5, print_every=1000):
    """
    Train the neural network on XOR dataset using backpropagation.
    
    Args:
        dataset: List of (x1, x2, target) tuples
        epochs: Number of training iterations through entire dataset
        learning_rate: Step size for gradient descent
        print_every: Print loss every N epochs
    """
    # Initialize network parameters
    params = {
        'w11': 0.30, 'w21': -0.20, 'b1': 0.00,  # Hidden neuron 1
        'w12': -0.40, 'w22': 0.10, 'b2': 0.00,  # Hidden neuron 2
        'v1': 0.20, 'v2': -0.30, 'c': 0.00      # Output neuron
    }
    
    print("Training XOR Neural Network")
    print("=" * 50)
    
    for epoch in range(epochs):
        # Train on each sample in the dataset
        for x1, x2, y in dataset:
            # Forward pass
            h1, h2, y_hat = forward_pass(x1, x2, **params)
            
            # Backward pass (compute gradients)
            gradients = backward_pass(x1, x2, y, h1, h2, y_hat, params['v1'], params['v2'])
            
            # Update weights
            params = update_weights(params, gradients, learning_rate)
        
        # Print progress
        if epoch % print_every == 0:
            loss = compute_loss(dataset, params)
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
    
    # Final loss
    final_loss = compute_loss(dataset, params)
    print(f"Epoch {epochs:5d} | Loss: {final_loss:.6f}")
    print("=" * 50)
    
    return params

def test_network(dataset, params):
    """Test the trained network on all inputs."""
    print("\nTesting Network on XOR Dataset:")
    print("-" * 50)
    for x1, x2, y in dataset:
        _, _, y_hat = forward_pass(x1, x2, **params)
        print(f"Input: ({x1}, {x2}) | Target: {y} | Prediction: {y_hat:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    # XOR dataset: (input1, input2, expected_output)
    xor_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    
    # Train the network
    trained_params = train_network(xor_dataset, epochs=10000, learning_rate=0.5)
    
    # Test the trained network
    test_network(xor_dataset, trained_params)
