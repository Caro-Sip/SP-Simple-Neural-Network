import numpy as np
import os
import cv2
import pickle
import nnfs
nnfs.init()

# =========================
# Layers
# =========================

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def predictions(self):
        return np.argmax(self.output, axis=1)


# =========================
# Loss
# =========================

class Loss_CCE:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct = y_pred[range(samples), y_true]
        return -np.log(correct)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples


class Softmax_CCE:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


# =========================
# Optimizer
# =========================

class Optimizer_Adam:
    def __init__(self, lr=0.001):
        self.lr = lr

    def update(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases


# =========================
# Model
# =========================

class Model:
    def __init__(self):
        self.layers = []
        self.loss = Loss_CCE()
        self.softmax_loss = Softmax_CCE()
        self.optimizer = Optimizer_Adam()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)
        return self.layers[-1].output

    def backward(self, output, y):
        self.softmax_loss.backward(output, y)
        self.layers[-1].dinputs = self.softmax_loss.dinputs

        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].backward(self.layers[i+1].dinputs)

    def train(self, X, y, epochs=5):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(self.loss.forward(output, y))
            self.backward(output, y)

            for layer in self.layers:
                if hasattr(layer, "weights"):
                    self.optimizer.update(layer)

            print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


# =========================
# Data
# =========================

def load_mnist(path, dataset):
    X, y = [], []
    for label in os.listdir(os.path.join(path, dataset)):
        for file in os.listdir(os.path.join(path, dataset, label)):
            img = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(img)
            y.append(int(label))
    X = np.array(X)
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    return X, np.array(y)


# =========================
# Save / Load
# =========================

def save_model(model, filename="model.pkl"):
    data = []
    for layer in model.layers:
        if hasattr(layer, "weights"):
            data.append((layer.weights, layer.biases))
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("✅ Model saved")


def load_model(model, filename="model.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    idx = 0
    for layer in model.layers:
        if hasattr(layer, "weights"):
            layer.weights, layer.biases = data[idx]
            idx += 1
    print("✅ Model loaded")


# =========================
# Menu Actions
# =========================

def build_model(input_size):
    model = Model()
    model.add(Layer_Dense(input_size, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())
    return model


def train_model():
    X, y = load_mnist("mnist_png", "train")
    model = build_model(X.shape[1])
    model.train(X, y, epochs=5)
    save_model(model)


def test_model():
    X, y = load_mnist("mnist_png", "test")
    model = build_model(X.shape[1])
    load_model(model)
    preds = model.predict(X[:10])

    for i in range(10):
        print(f"Sample {i+1}: Predicted={preds[i]}, Actual={y[i]}")


# =========================
# Menu
# =========================

def main():
    while True:
        print("\n=== MNIST Neural Network ===")
        print("1. Train model")
        print("2. Test model")
        print("3. Exit")

        choice = input("Choose (1-3): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            print("👋 Exiting")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
