import numpy as np
import os
from PIL import Image
import tkinter as tk
import pickle
import matplotlib.pyplot as plt

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def load_dataset(path, num_per_digit=10, return_paths=False):
    X = []
    y = []
    paths = []

    for label in range(10):
        folder = f"{path}/{label}"
        files = os.listdir(folder)[:num_per_digit]
        for file in files:
            img_path = f"{folder}/{file}"
            img = Image.open(img_path).convert("L")
            img = img.resize((28, 28))
            img = np.array(img) / 255.0
            X.append(img.flatten())
            y.append(label)
            paths.append(img_path)

    if return_paths:
        return np.array(X), np.array(y), paths
    return np.array(X), np.array(y)

def relu(Z):
    return np.maximum(0, Z)

def save_weights(W1, b1, W2, b2, W3, b3, filename='model_weights.pkl'):
    weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")

def load_weights(filename='model_weights.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        print(f"Weights loaded from {filename}")
        return weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3']
    else:
        print(f"No saved weights found at {filename}")
        return None

def init_model():
    W1 = np.random.randn(784, 128) * 0.01
    b1 = np.zeros((1, 128))
    W2 = np.random.randn(128, 64) * 0.01
    b2 = np.zeros((1, 64))
    W3 = np.random.randn(64, 10) * 0.01
    b3 = np.zeros((1, 10))
    return W1, b1, W2, b2, W3, b3

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def cross_entropy(A3, y):
    return -np.log(A3[range(len(y)), y]).mean()

def backward(X, y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = len(y)
    dZ3 = A3.copy()
    dZ3[range(m), y] -= 1
    dZ3 /= m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3
def train_model(X, y, epochs=1000, lr=0.01):
    W1, b1, W2, b2, W3, b3 = init_model()
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward(X, W1, b1, W2, b2, W3, b3)
        loss = cross_entropy(A3, y)
        dW1, db1, dW2, db2, dW3, db3 = backward(X, y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    save_weights(W1, b1, W2, b2, W3, b3)
    return W1, b1, W2, b2, W3, b3
def evaluate_model(X, y, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward(X, W1, b1, W2, b2, W3, b3)
    preds = np.argmax(A3, axis=1)
    accuracy = np.mean(preds == y)
    print(f"Accuracy: {accuracy*100:.2f}%")
    return accuracy
def create_gui(W1, b1, W2, b2, W3, b3):
    canvas_data = np.zeros((100, 100))
    
    def paint(event):
        x, y = event.x, event.y
        canvas.create_oval(x, y, x+5, y+5, fill="white")
        if 0 <= y < 100 and 0 <= x < 100:
            canvas_data[min(y, 95):min(y+5, 100), min(x, 95):min(x+5, 100)] = 1
    
    def predict_digit():
        X = canvas_data.reshape(1, -1)
        _, _, _, _, _, A3 = forward(X, W1, b1, W2, b2, W3, b3)
        print("Prediction:", np.argmax(A3))
    
    def clear_canvas():
        canvas.delete("all")
        canvas_data[:] = 0
    
    root = tk.Tk()
    root.title("Digit Recognizer")
    canvas = tk.Canvas(root, width=100, height=100, bg="black")
    canvas.pack()
    canvas.bind("<B1-Motion>", paint)
    
    btn_predict = tk.Button(root, text="Predict", command=predict_digit)
    btn_predict.pack()
    
    btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
    btn_clear.pack()
    
    root.mainloop()

if __name__ == "__main__":
    print("=" * 50)
    print("MNIST Digit Recognition")
    print("=" * 50)
    
    choice = input("\nChoose mode:\n1. Train model\n2. Test model\n\nEnter choice (1 or 2): ")
    
    if choice == "1":
        print("\nLoading training data (10 images per digit)...")
        X_train, y_train = load_dataset("mnist_png/train", num_per_digit=10)
        print(f"Loaded {len(X_train)} training images")
        
        print("\nTraining model...")
        W1, b1, W2, b2, W3, b3 = train_model(X_train, y_train, epochs=1000, lr=0.5)
        
        print("\nTraining complete!")
        
    elif choice == "2":
        print("\nLoading saved weights...")
        weights = load_weights()
        
        if weights is None:
            print("No trained model found. Please train first.")
        else:
            W1, b1, W2, b2, W3, b3 = weights
            
            print("\nLoading test data (10 images total)...")
            X_test, y_test, img_paths = load_dataset("mnist_png/test", num_per_digit=1, return_paths=True)
            print(f"Loaded {len(X_test)} test images")
            
            print("\nEvaluating model...")
            evaluate_model(X_test, y_test, W1, b1, W2, b2, W3, b3)
            
            print("\nShowing predictions for each test image:")
            print("Close the image window to see the next one...\n")
            
            for i in range(len(X_test)):
                X_single = X_test[i:i+1]
                _, _, _, _, _, A3 = forward(X_single, W1, b1, W2, b2, W3, b3)
                pred = np.argmax(A3)
                actual = y_test[i]
                status = "✓" if pred == actual else "✗"
                
                img = Image.open(img_paths[i])
                plt.figure(figsize=(4, 4))
                plt.imshow(img, cmap='gray')
                plt.title(f"Predicted: {pred} | Actual: {actual} {status}", fontsize=14)
                plt.axis('off')
                plt.show()
                
                print(f"Image {i+1}: Predicted={pred}, Actual={actual} {status}")
    
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")