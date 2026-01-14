import struct
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(Ws, bs, x, activation=sigmoid):
    x = x.astype(np.float32).reshape(-1, 1)
    assert Ws[0].shape[1] == x.size, f"Mismatch: first W expects {Ws[0].shape[1]} inputs, got {x.size}"
    for W, b in zip(Ws, bs):
        assert W.shape[1] == x.size, f"Mismatch: W expects {W.shape[1]} inputs, got {x.size}"
        if b is None:
            b = np.zeros((W.shape[0], 1), dtype=np.float32)
        x = activation(W.dot(x) + b)
    return x

def softmax(z):
    """Compute softmax probabilities from logits."""
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / exp.sum()

def read_first_image(image_path, label_path=None):
    """Read the first image from MNIST dataset files."""
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        buf = f.read(rows * cols)  # read first image only
        img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        img = img.reshape(rows, cols)

    # read the label if provided
    label = None
    if label_path is not None:
        with open(label_path, 'rb') as lf:
            lm, ln = struct.unpack('>II', lf.read(8))
            lb = lf.read(1)
            if lb:
                label = int.from_bytes(lb, 'big')

    return img, label


if __name__ == '__main__':
    # Read MNIST image and label
    img, label = read_first_image('mnist_data/train-images-idx3-ubyte',
                                   'mnist_data/train-labels-idx1-ubyte')

    # show basic shape
    print('image shape =', img.shape)

    # optional display
    try:
        plt.imshow(img, cmap='gray')
        plt.title(f'Training image (label={label})')
        plt.axis('off')
        plt.show()
    except Exception:
        pass

    # prepare input vector for feedforward: normalize and flatten
    x = (img.flatten() / 255.0).astype(np.float32)

    # simple network: 784 -> 64 -> 16 -> 10
    np.random.seed(50)
    input_size = x.size
    sizes = [input_size, 64, 16, 10]

    # random weights and biases
    Ws = [np.random.randn(sizes[i+1], sizes[i]).astype(np.float32) for i in range(len(sizes)-1)]
    bs = [np.random.randn(sizes[i+1], 1).astype(np.float32) for i in range(len(sizes)-1)]

    # prepare input vector and compute hidden activations using feedforward
    x_vec = x.reshape(-1, 1).astype(np.float32)

    # use `feedforward` for all layers except the final one so we can
    # compute final logits here and apply softmax
    if len(Ws) > 1:
        x_vec = feedforward(Ws[:-1], bs[:-1], x_vec)

    # final layer (logits) -- linear, no activation
    W_last, b_last = Ws[-1], bs[-1]
    logits = (W_last.dot(x_vec) + b_last).astype(np.float64).ravel()

    # print logits before softmax normalization
    print('\nlogits (pre-softmax):')
    for i, logit in enumerate(logits):
        print(f"{i}: {logit:.6f}")

    # softmax -> probabilities in [0,1] summing to 1
    probs = softmax(logits)
    pred = int(probs.argmax())

    # print each class probability on its own line (0..9)
    print('\nclass probabilities (post-softmax):')
    for i, p in enumerate(probs):
        print(f"{i}: {p:.6f}")
    print('predicted class =', pred)
    if label is not None:
        print('true label =', label)