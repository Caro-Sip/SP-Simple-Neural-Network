import numpy as np

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

# example: network 178 -> 64 -> 16 -> 10
np.random.seed(42)
sizes = [178, 64, 16, 10]
Ws = [np.random.randn(sizes[i+1], sizes[i]).astype(np.float32) for i in range(len(sizes)-1)]
bs = [np.random.randn(sizes[i+1], 1).astype(np.float32) for i in range(len(sizes)-1)]

# example input (must match first layer size = 178)
x = np.random.rand(sizes[0], 1).astype(np.float32)

output = feedforward(Ws, bs, x)
print("output.shape =", output.shape)
print(output)