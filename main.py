import math

def sigmoid(h):
    return 1/(1 + (math.exp(-h)))

def neuron(x1, x2, w1, w2, b):
    # weighted sum
    h = x1*w1 + x2*w2 + b
    # activation function
    return sigmoid(h)

def network(x1, x2,
            w11, w21, b1,
            w12, w22, b2,
            v1, v2, c):
    # hidden neuron 1
    h1 = neuron(x1, x2, w11, w21, b1)
    # hidden neuron 2
    h2 = neuron(x1, x2, w12, w22, b2)
    # output neuron
    y_hat = neuron(h1, h2, v1, v2, c)

    # y_hat is our prediction, but we're also returning the
    # hidden neuron outputs as well. These can be ignored
    # if you just want the prediction - these are more
    # useful when we get to training.
    return h1, h2, y_hat
# Input dataset
xor_dataset = [ (0,0,0), (0,1,1), (1,0,1), (1,1,0) ]

# Starting parameters
w11, w21, b1 = 0.30, -0.20, 0.00
w12, w22, b2 = -0.40,  0.10, 0.00
v1, v2, c = 0.20, -0.30, 0.00


def predict_all(dataset, w11, w21, b1, w12, w22, b2, v1, v2, c):
    for (x1, x2, y) in dataset:
        _, _, y_hat = network(x1, x2, w11, w21, b1, w12, w22, b2, v1, v2, c)
        print(f"input=({x1},{x2})  target={y}  y_hat={y_hat:.4f}")


def train_on_sample(x1, x2, y,
                    w11, w21, b1,
                    w12, w22, b2,
                    v1, v2, c,
                    eta=0.1):
    # Forward pass
    h1, h2, y_hat = network(x1, x2, w11, w21, b1, w12, w22, b2, v1, v2, c)

    delta_out = (y_hat - y) * y_hat * (1 - y_hat)

    # Gradients for output weights
    dv1 = delta_out * h1
    dv2 = delta_out * h2
    dc  = delta_out * 1

    # Update output weights and bias
    v1 -= eta * dv1
    v2 -= eta * dv2
    c  -= eta * dc

    # Hidden layer error terms
    delta_h1 = delta_out * v1 * h1 * (1 - h1)
    delta_h2 = delta_out * v2 * h2 * (1 - h2)

    # Gradients for hidden weights
    dw11 = delta_h1 * x1
    dw21 = delta_h1 * x2
    db1  = delta_h1 * 1

    dw12 = delta_h2 * x1
    dw22 = delta_h2 * x2
    db2  = delta_h2 * 1

    # Update hidden weights
    w11 -= eta * dw11
    w21 -= eta * dw21
    b1  -= eta * db1

    w12 -= eta * dw12
    w22 -= eta * dw22
    b2  -= eta * db2

    return (w11, w21, b1,
            w12, w22, b2,
            v1, v2, c)


if __name__ == '__main__':
    # Print predictions with initial parameters
    predict_all(xor_dataset, w11, w21, b1, w12, w22, b2, v1, v2, c)

    # Perform one training update on the first sample (preserves original behavior)
    (x1, x2, y) = xor_dataset[0]
    w11, w21, b1, w12, w22, b2, v1, v2, c = train_on_sample(
        x1, x2, y,
        w11, w21, b1,
        w12, w22, b2,
        v1, v2, c,
        eta=0.1
    )