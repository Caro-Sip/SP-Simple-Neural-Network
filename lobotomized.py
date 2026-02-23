# ====================================================================
#                            IMPORTS & SETUP
# ====================================================================

import numpy as np
import nnfs
import os
import cv2
import pickle
import random
nnfs.init()

# Dependencies: pip install numpy nnfs opencv-python matplotlib
# Files needed: mnist_model.pkl (trained model data)


# ====================================================================
#                    NEURAL NETWORK COMPONENTS
# ====================================================================

# -------------------- Layers --------------------

class Layer_Dense:
    """
    A Fully Connected (Dense) Layer.
    Every input neuron connects to every output neuron.
    Learns by adjusting 'weights' (importance of each input)
    and 'biases' (baseline offset per neuron).
    """
    def __init__(self, num_inputs, num_neurons):
        # Small random starting weights so outputs don't explode early on
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        # Biases start at zero — one per neuron
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs, training):
        # Save inputs — we'll need them during backprop
        self.inputs = inputs
        # Core formula: output = inputs × weights + bias
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, gradients_from_next_layer):
        # How much each weight contributed to the error
        self.weight_gradients = np.dot(self.inputs.T, gradients_from_next_layer)
        # How much each bias contributed to the error
        self.bias_gradients   = np.sum(gradients_from_next_layer, axis=0, keepdims=True)
        # How much each input contributed (passed back to the previous layer)
        self.input_gradients  = np.dot(gradients_from_next_layer, self.weights.T)


class Layer_Input:
    """
    A pass-through entry point for raw data.
    Doesn't transform anything — just holds the input so
    the first real layer has a 'prev.output' to read from.
    """
    def forward(self, inputs, training):
        self.output = inputs


# -------------------- Activations --------------------

class Activation_ReLU:
    """
    ReLU (Rectified Linear Unit) Activation.
    Rule: if a value is negative → replace with 0. Otherwise keep it.
    Formula: output = max(0, input)

    Why it exists: without a non-linear activation, stacking multiple
    layers is mathematically the same as having just one layer.
    ReLU lets neurons 'switch off' when they're not relevant.
    """
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, gradients_from_next_layer):
        self.input_gradients = gradients_from_next_layer.copy()
        # Neurons that were off (input ≤ 0) didn't contribute — zero their gradient
        self.input_gradients[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    """
    Softmax Activation.
    Converts raw scores into probabilities that all add up to 1.0 (100%).
    Used as the final layer in classification networks.

    Example: raw scores [2.1, 0.5, 3.8] → probabilities [0.17, 0.07, 0.76]
    Now you can say "76% confident this is class 2."

    Uses exp() to make all values positive, then divides by the total
    so they sum to 1. Subtracting the max first prevents numerical overflow.
    """
    def forward(self, inputs, training):
        self.inputs = inputs
        # Subtract max per row for numerical stability (prevents exp() overflow)
        exp_values    = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities

    def backward(self, gradients_from_next_layer):
        self.input_gradients = np.empty_like(gradients_from_next_layer)
        for i, (single_output, single_gradient) in \
                enumerate(zip(self.output, gradients_from_next_layer)):
            single_output = single_output.reshape(-1, 1)
            # Jacobian: how each output probability affects every other
            jacobian_matrix = (np.diagflat(single_output)
                               - np.dot(single_output, single_output.T))
            self.input_gradients[i] = np.dot(jacobian_matrix, single_gradient)

    def predictions(self, outputs):
        # Predicted class = the digit with the highest probability
        return np.argmax(outputs, axis=1)


# -------------------- Loss Functions --------------------

class Loss:
    """
    Base class for loss (error) functions.
    Averages the per-sample loss into one number.
    A lower loss = the network is doing better.
    """
    def calculate(self, predicted_output, true_labels):
        per_sample_losses = self.forward(predicted_output, true_labels)
        average_loss      = np.mean(per_sample_losses)
        return average_loss


class Loss_CategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy Loss.
    Measures how wrong the network's probability predictions are
    for a classification problem (e.g. which digit 0-9 is this?).

    Formula: loss = -log(probability assigned to the correct class)

    -log(0.99) ≈ 0.01  → very confident AND correct  → tiny penalty
    -log(0.50) ≈ 0.69  → uncertain                   → medium penalty
    -log(0.01) ≈ 4.60  → very confident BUT WRONG     → huge penalty

    The log curve means being confidently wrong is punished especially hard.
    """
    def forward(self, predicted_probs, true_labels):
        num_samples = len(predicted_probs)
        # Clip to avoid log(0) which is mathematically undefined
        clipped_probs = np.clip(predicted_probs, 1e-7, 1 - 1e-7)

        if len(true_labels.shape) == 1:
            # true_labels are class indices e.g. [3, 7, 2, ...]
            correct_class_probs = clipped_probs[range(num_samples), true_labels]
        elif len(true_labels.shape) == 2:
            # true_labels are one-hot vectors e.g. [[0,0,0,1,...], ...]
            correct_class_probs = np.sum(clipped_probs * true_labels, axis=1)

        losses = -np.log(correct_class_probs)
        return losses

    def backward(self, predicted_probs, true_labels):
        num_samples = len(predicted_probs)
        num_classes = len(predicted_probs[0])

        if len(true_labels.shape) == 1:
            # Convert class indices to one-hot vectors for the gradient formula
            true_labels = np.eye(num_classes)[true_labels]

        self.input_gradients = -true_labels / predicted_probs
        # Normalize by sample count so gradient magnitude doesn't grow with batch size
        self.input_gradients = self.input_gradients / num_samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combined Softmax + Cross-Entropy backward pass shortcut.

    Mathematically, when you work out the gradient of
    (Softmax followed by CrossEntropyLoss), it simplifies to:
        gradient = predicted_probabilities - one_hot(true_label)

    This is much simpler and more numerically stable than computing
    the two backward passes separately. Only used during backpropagation.
    """
    def backward(self, predicted_probs, true_labels):
        num_samples = len(predicted_probs)
        if len(true_labels.shape) == 2:
            true_labels = np.argmax(true_labels, axis=1)

        self.input_gradients = predicted_probs.copy()
        # Subtract 1 from the probability of the correct class
        self.input_gradients[range(num_samples), true_labels] -= 1
        # Normalize by sample count
        self.input_gradients = self.input_gradients / num_samples


# -------------------- Optimizer --------------------

class Optimizer_SGD:
    """
    SGD (Stochastic Gradient Descent) Optimizer.
    After backprop computes gradients (how much each weight caused the error),
    SGD nudges every weight slightly in the direction that reduces the error.

    Formula: new_weight = old_weight - learning_rate × weight_gradient

    learning_rate controls the step size:
    - Too large  → overshoots the minimum, training becomes unstable
    - Too small  → takes forever to learn

    'Stochastic' means we update after each mini-batch rather than
    waiting to process the entire dataset first.
    """
    def __init__(self, learning_rate=0.1):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate

    def pre_update_params(self):
        pass  # Hook for features like learning rate decay

    def update_params(self, layer):
        # Nudge weights and biases down the gradient slope
        layer.weights -= self.learning_rate * layer.weight_gradients
        layer.biases  -= self.learning_rate * layer.bias_gradients

    def post_update_params(self):
        pass  # Hook for tracking iteration count, etc.


# -------------------- Accuracy --------------------

class Accuracy_Categorical:
    """
    Tracks what fraction of predictions match the true labels.
    Note: accuracy and loss are related but different.
    A model can improve its loss (confidence calibration) without
    changing its accuracy (which class it picks).
    """
    def calculate(self, predicted_classes, true_labels):
        if len(true_labels.shape) == 2:
            true_labels = np.argmax(true_labels, axis=1)
        correct    = predicted_classes == true_labels
        accuracy   = np.mean(correct)
        return accuracy


# -------------------- Model --------------------

class Model:
    """
    The top-level container. Owns all layers and orchestrates:
    - forward()   : make a prediction by passing data through every layer
    - backward()  : propagate error gradients back through every layer
    - train()     : repeat forward → measure loss → backward → update weights
    """
    def __init__(self):
        self.layers = []
        self.softmax_loss_shortcut = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss      = loss
        self.optimizer = optimizer
        self.accuracy  = accuracy

    def finalize(self):
        """
        Wire each layer to its previous and next layer.
        forward pass : each layer reads  prev.output
        backward pass: each layer reads  next.input_gradients
        """
        self.input_layer   = Layer_Input()
        num_layers         = len(self.layers)
        self.trainable_layers = []

        for i in range(num_layers):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < num_layers - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Use the faster combined backward pass if Softmax + CrossEntropy are paired
        if (isinstance(self.layers[-1], Activation_Softmax) and
                isinstance(self.loss, Loss_CategoricalCrossentropy)):
            self.softmax_loss_shortcut = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):
        num_steps = 1

        if validation_data is not None:
            X_val, y_val = validation_data

        if batch_size is not None:
            num_steps = len(X) // batch_size
            if num_steps * batch_size < len(X):
                num_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            for step in range(num_steps):
                # Slice the current mini-batch
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # 1. Forward pass — make predictions
                output = self.forward(batch_X, training=True)

                # 2. Measure how wrong the predictions are
                loss = self.loss.calculate(output, batch_y)

                # 3. Get the predicted class index per sample
                predicted_classes = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predicted_classes, batch_y)

                # 4. Backward pass — compute gradients for every weight
                self.backward(output, batch_y)

                # 5. Update all weights using their gradients
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == num_steps - 1:
                    print(f'  step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f}')

            # Evaluate on validation set (no weight updates here)
            if validation_data is not None:
                val_output   = self.forward(X_val, training=False)
                val_loss     = self.loss.calculate(val_output, y_val)
                val_classes  = self.output_layer_activation.predictions(val_output)
                val_accuracy = self.accuracy.calculate(val_classes, y_val)
                print(f'  validation, acc: {val_accuracy:.3f}, loss: {val_loss:.3f}')

    def forward(self, X, training):
        """Pass data through every layer left to right."""
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output  # output of the final layer

    def backward(self, output, y):
        """
        Propagate error gradients right to left through every layer.
        Uses the Softmax+CrossEntropy shortcut when available.
        """
        if self.softmax_loss_shortcut is not None:
            self.softmax_loss_shortcut.backward(output, y)
            # Inject the shortcut's gradient into the last layer
            self.layers[-1].input_gradients = \
                self.softmax_loss_shortcut.input_gradients
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.input_gradients)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.input_gradients)


# ====================================================================
#                  IMAGE PREPROCESSING & AUGMENTATION
# ====================================================================

def preprocess_drawn_image(canvas):
    """
    Convert a user-drawn image into MNIST format:
    1. Invert colors (MNIST = white digit on black background)
    2. Find the digit's bounding box
    3. Crop, make square, resize to 20x20
    4. Center the digit by its center of mass inside a 28x28 frame
    """
    # We draw black on white; MNIST is white on black — flip it
    inverted = 255 - canvas

    # Threshold to a clean binary image for bounding box detection
    _, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)

    nonzero_pixels = cv2.findNonZero(binary)
    if nonzero_pixels is None:
        return np.zeros((28, 28), dtype=np.uint8)  # nothing drawn

    x, y, w, h = cv2.boundingRect(nonzero_pixels)

    # Add a small border so the digit doesn't touch the edges
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(canvas.shape[1] - x, w + 2 * pad)
    h = min(canvas.shape[0] - y, h + 2 * pad)

    cropped = inverted[y:y + h, x:x + w]

    # Pad shorter dimension to make it square (avoids distortion on resize)
    if h > w:
        diff  = h - w
        left  = diff // 2
        right = diff - left
        cropped = cv2.copyMakeBorder(cropped, 0, 0, left, right,
                                     cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        diff   = w - h
        top    = diff // 2
        bottom = diff - top
        cropped = cv2.copyMakeBorder(cropped, top, bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=0)

    # MNIST digits sit in a ~20x20 area inside the 28x28 frame
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)

    # Use center of mass to position the digit properly in the 28x28 frame
    moments = cv2.moments(resized)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x, center_y = 10, 10

    output_frame = np.zeros((28, 28), dtype=np.uint8)
    # Offset so the digit's center of mass lands near the middle of the 28x28 frame
    offset_x = max(0, min(8, 14 - center_x))
    offset_y = max(0, min(8, 14 - center_y))
    output_frame[offset_y:offset_y + 20, offset_x:offset_x + 20] = resized

    return output_frame


def augment_image(img):
    """
    Apply random transformations to a training image.
    More variety during training → model generalizes better to real handwriting.
    """
    h, w   = img.shape[:2]
    result = img.copy()

    # Random rotation — a slightly tilted digit is still the same digit
    if random.random() < 0.7:
        angle            = random.uniform(-15, 15)
        rotation_matrix  = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result           = cv2.warpAffine(result, rotation_matrix, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)

    # Random translation — digit isn't always perfectly centered
    if random.random() < 0.7:
        shift_x      = random.uniform(-2, 2)
        shift_y      = random.uniform(-2, 2)
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        result       = cv2.warpAffine(result, shift_matrix, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)

    # Random scale — digit might be written larger or smaller
    if random.random() < 0.5:
        scale       = random.uniform(0.9, 1.1)
        zoom_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        result      = cv2.warpAffine(result, zoom_matrix, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)

    # Random shear — a slight slant in the stroke
    if random.random() < 0.3:
        shear        = random.uniform(-0.1, 0.1)
        shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
        result       = cv2.warpAffine(result, shear_matrix, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)

    # Gaussian noise — simulates scanner noise or shaky pen strokes
    if random.random() < 0.3:
        noise  = np.random.normal(0, random.uniform(5, 20),
                                  result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Random brightness / contrast
    if random.random() < 0.3:
        contrast   = random.uniform(0.8, 1.2)
        brightness = random.uniform(-20, 20)
        result     = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)

    # Erosion (thinner strokes) or dilation (thicker strokes)
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            result = cv2.erode(result, kernel, iterations=1)
        else:
            result = cv2.dilate(result, kernel, iterations=1)

    return result


# ====================================================================
#                      DATA LOADING & UTILITIES
# ====================================================================

def load_mnist_dataset(dataset_split, base_path):
    """
    Load images from the mnist_png folder structure:
      mnist_png/train/0/*.png
      mnist_png/train/1/*.png  ...etc
    """
    digit_labels = os.listdir(os.path.join(base_path, dataset_split))
    images     = []
    labels     = []
    file_paths = []

    for digit_label in digit_labels:
        folder = os.path.join(base_path, dataset_split, digit_label)
        for filename in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
            images.append(image)
            labels.append(digit_label)
            file_paths.append(f"{dataset_split}/{digit_label}/{filename}")

    return np.array(images), np.array(labels).astype('uint8'), file_paths


def create_data_mnist(path):
    train_images, train_labels, train_paths = load_mnist_dataset('train', path)
    test_images,  test_labels,  test_paths  = load_mnist_dataset('test',  path)
    return train_images, train_labels, test_images, test_labels, train_paths, test_paths


# ====================================================================
#                      MODEL SAVE/LOAD UTILITIES
# ====================================================================

def save_model(model, filename='mnist_model.pkl'):
    """Save the trained weights and biases to disk."""
    model_data = {'layers': []}
    for layer in model.trainable_layers:
        model_data['layers'].append({
            'weights': layer.weights,
            'biases':  layer.biases
        })
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[OK] Model saved to '{filename}'!")


def load_model(model, filename='mnist_model.pkl'):
    """Load previously saved weights and biases from disk."""
    if not os.path.exists(filename):
        print(f"\n[ERROR] Model file '{filename}' not found!")
        print("Please train the model first (Option 1).")
        return False
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    for i, layer in enumerate(model.trainable_layers):
        layer.weights = model_data['layers'][i]['weights']
        layer.biases  = model_data['layers'][i]['biases']
    print(f"\n[OK] Model loaded from '{filename}'!")
    return True


# ====================================================================
#                         TRAINING FUNCTION
# ====================================================================

def train_model():
    print("\n" + "=" * 50)
    print("TRAINING MODE (with augmentation)")
    print("=" * 50)
    print("\n[INFO] Loading MNIST dataset...")

    train_images, train_labels, test_images, test_labels, _, _ = \
        create_data_mnist('mnist_png')

    # Shuffle so batches aren't all the same digit
    shuffle_order = np.array(range(train_images.shape[0]))
    np.random.shuffle(shuffle_order)
    train_images = train_images[shuffle_order]
    train_labels = train_labels[shuffle_order]

    # Normalize test images: pixel 0–255 → roughly -1.0 to +1.0
    test_images_normalized = (test_images.reshape(test_images.shape[0], -1)
                               .astype(np.float32) - 127.5) / 127.5

    print(f"[OK] Loaded {len(train_images)} training and "
          f"{len(test_images)} test images!")

    model = Model()
    model.add(Layer_Dense(784, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 10))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(learning_rate=0.05),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()

    print("\n[INFO] Starting training with augmentation...")
    print("(Each epoch re-augments the data for variety)")
    print("-" * 50)

    num_epochs = 10
    batch_size = 128

    for epoch in range(1, num_epochs + 1):
        print(f'\n=== Epoch {epoch}/{num_epochs} (augmenting data...) ===')

        # Augment training images fresh each epoch
        augmented_images = np.empty_like(train_images)
        for i in range(train_images.shape[0]):
            if random.random() < 0.8:
                augmented_images[i] = augment_image(train_images[i])
            else:
                augmented_images[i] = train_images[i]

        # Flatten 28x28 → 784 and normalize 0–255 → -1 to +1
        augmented_flat = (augmented_images.reshape(augmented_images.shape[0], -1)
                          .astype(np.float32) - 127.5) / 127.5

        model.train(
            augmented_flat, train_labels,
            validation_data=(test_images_normalized, test_labels),
            epochs=1,
            batch_size=batch_size,
            print_every=100
        )

    save_model(model)
    print("\n[OK] Training complete!")
    input("\nPress Enter to return to menu...")


# ====================================================================
#                    UI & INTERACTIVE TESTING
# ====================================================================

def interpolate_points(p1, p2, num_points=10):
    """Generate intermediate points between two points for smoother lines."""
    x1, y1 = p1
    x2, y2 = p2
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        points.append((x, y))
    return points


def draw_smooth_line(canvas, p1, p2, brush_size=12):
    """Draw a smooth anti-aliased line between two points."""
    # Calculate distance between points for adaptive interpolation
    dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    # More interpolation points for smoother lines during fast movement
    num_points = max(int(dist), 1)
    
    points = interpolate_points(p1, p2, num_points)
    
    # Draw circles at each point for consistent thickness
    for point in points:
        cv2.circle(canvas, point, brush_size, 0, -1, lineType=cv2.LINE_AA)
    
    # Draw main anti-aliased line
    cv2.line(canvas, p1, p2, 0, brush_size * 2, lineType=cv2.LINE_AA)
    return canvas


def test_model():
    print("\n" + "=" * 50)
    print("TESTING MODE — DRAW YOUR OWN DIGIT")
    print("=" * 50)

    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.patches import FancyBboxPatch, Rectangle
        import matplotlib.patches as mpatches
    except ImportError:
        print("\n[ERROR] matplotlib is required for drawing mode.")
        print("Install it with: pip install matplotlib")
        input("\nPress Enter to return to menu...")
        return

    model = Model()
    model.add(Layer_Dense(784, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 10))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(learning_rate=0.05),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()

    if not load_model(model):
        input("\nPress Enter to return to menu...")
        return

    # --- UI Configuration ---
    canvas_size = 280
    brush_size = 16
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    is_drawing = False
    background = None  # For blitting optimization
    last_point = None
    stroke_points = []  # Store points for smooth drawing
    
    # Color scheme
    bg_color = '#1a1a2e'
    panel_color = '#16213e'
    accent_color = '#0f3460'
    highlight_color = '#e94560'
    text_color = '#eaeaea'
    success_color = '#00d26a'
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 7), facecolor=bg_color)
    fig.canvas.manager.set_window_title('MNIST Digit Recognition')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 0.75, 0.15], 
                          width_ratios=[0.05, 0.4, 0.4, 0.15],
                          hspace=0.15, wspace=0.1)
    
    # Title area
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(bg_color)
    ax_title.axis('off')
    ax_title.text(0.5, 0.65, 'Neural Network Digit Recognition', 
                  fontsize=22, fontweight='bold', color=text_color,
                  ha='center', va='center', transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15, 'Draw a digit (0-9) and let the AI predict it', 
                  fontsize=12, color='#888888', ha='center', va='center',
                  transform=ax_title.transAxes)
    
    # Canvas area
    ax_canvas = fig.add_subplot(gs[1, 1])
    ax_canvas.set_facecolor(panel_color)
    ax_canvas.set_title('Drawing Canvas', fontsize=14, color=text_color, pad=10)
    canvas_display = ax_canvas.imshow(canvas, cmap='gray', vmin=0, vmax=255,
                                       aspect='equal', interpolation='bilinear')
    ax_canvas.axis('off')
    
    # Add border around canvas
    for spine in ax_canvas.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(2)
    
    # Result area
    ax_result = fig.add_subplot(gs[1, 2])
    ax_result.set_facecolor(panel_color)
    # ax_result.set_title('Prediction Result', fontsize=14, color=text_color, pad=10)
    ax_result.axis('off')
    
    # Initial placeholder text
    result_text = ax_result.text(
        0.5, 0.5, 'Draw a digit\nand click\nPredict',
        ha='center', va='center', fontsize=18, color='#666666',
        transform=ax_result.transAxes, style='italic'
    )
    
    # Confidence bars area
    ax_bars = fig.add_subplot(gs[1, 3])
    ax_bars.set_facecolor(panel_color)
    ax_bars.set_title('Confidence', fontsize=12, color=text_color, pad=10)
    ax_bars.set_xlim(0, 1)
    ax_bars.set_ylim(-0.5, 9.5)
    ax_bars.set_yticks(range(10))
    ax_bars.set_yticklabels([str(i) for i in range(10)], color=text_color, fontsize=11)
    ax_bars.set_xticks([])
    ax_bars.invert_yaxis()
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.spines['bottom'].set_visible(False)
    ax_bars.spines['left'].set_color(accent_color)
    
    # Initialize empty bars
    bars = ax_bars.barh(range(10), [0]*10, color=accent_color, height=0.6)
    bar_labels = []
    for i in range(10):
        label = ax_bars.text(0.02, i, '', va='center', ha='left', 
                            color=text_color, fontsize=9, fontweight='bold')
        bar_labels.append(label)
    
    # Button area
    ax_buttons = fig.add_subplot(gs[2, 1:3])
    ax_buttons.set_facecolor(bg_color)
    ax_buttons.axis('off')
    
    # Custom styled buttons
    button_width = 0.12
    button_height = 0.5
    button_y = 0.25
    
    ax_btn_predict = fig.add_axes([0.25, 0.08, 0.15, 0.05])
    ax_btn_clear = fig.add_axes([0.42, 0.08, 0.15, 0.05])
    ax_btn_exit = fig.add_axes([0.59, 0.08, 0.15, 0.05])
    
    btn_predict = Button(ax_btn_predict, 'Predict', color=success_color, hovercolor='#00b359')
    btn_clear = Button(ax_btn_clear, 'Clear', color=accent_color, hovercolor='#1a4a7a')
    btn_exit = Button(ax_btn_exit, 'Exit', color=highlight_color, hovercolor='#c73750')
    
    # Style button text
    btn_predict.label.set_fontsize(12)
    btn_predict.label.set_fontweight('bold')
    btn_predict.label.set_color('white')
    btn_clear.label.set_fontsize(12)
    btn_clear.label.set_fontweight('bold')
    btn_clear.label.set_color('white')
    btn_exit.label.set_fontsize(12)
    btn_exit.label.set_fontweight('bold')
    btn_exit.label.set_color('white')
    
    # Instructions text
    ax_instructions = fig.add_axes([0.05, 0.02, 0.15, 0.05])
    ax_instructions.axis('off')
    ax_instructions.text(0, 0.5, 'Hold left mouse button to draw', 
                        fontsize=9, color='#666666', va='center')

    def on_mouse_press(event):
        nonlocal is_drawing, last_point, stroke_points, background
        if event.inaxes == ax_canvas and event.button == 1:
            is_drawing = True
            x, y = int(event.xdata), int(event.ydata)
            last_point = (x, y)
            stroke_points = [(x, y)]
            # Capture background for blitting
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(ax_canvas.bbox)
            # Draw initial point
            cv2.circle(canvas, (x, y), brush_size, 0, -1, lineType=cv2.LINE_AA)
            canvas_display.set_data(canvas)
            ax_canvas.draw_artist(canvas_display)
            fig.canvas.blit(ax_canvas.bbox)

    def on_mouse_release(event):
        nonlocal is_drawing, last_point, stroke_points, canvas, background
        if is_drawing:
            # Apply slight Gaussian blur for smoother edges
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            canvas_display.set_data(canvas)
            fig.canvas.draw_idle()
        is_drawing = False
        last_point = None
        stroke_points = []
        background = None

    def on_mouse_move(event):
        nonlocal canvas, last_point, stroke_points, background
        if is_drawing and event.inaxes == ax_canvas and event.xdata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                current_point = (x, y)
                
                if last_point is not None:
                    # Draw smooth line between points
                    draw_smooth_line(canvas, last_point, current_point, brush_size)
                
                stroke_points.append(current_point)
                last_point = current_point
                
                # Use blitting for faster rendering
                canvas_display.set_data(canvas)
                if background is not None:
                    fig.canvas.restore_region(background)
                ax_canvas.draw_artist(canvas_display)
                fig.canvas.blit(ax_canvas.bbox)

    def update_confidence_bars(output):
        """Update the confidence bar chart."""
        probs = output[0]
        max_idx = np.argmax(probs)
        
        for i, (bar, prob, label) in enumerate(zip(bars, probs, bar_labels)):
            bar.set_width(prob)
            
            # Color scheme: highlight the predicted digit
            if i == max_idx:
                bar.set_color(success_color)
            elif prob > 0.1:
                bar.set_color(highlight_color)
            else:
                bar.set_color(accent_color)
            
            # Update percentage label
            if prob > 0.05:
                label.set_text(f'{prob*100:.0f}%')
                label.set_x(prob + 0.02)
            else:
                label.set_text('')

    def on_predict_clicked(event):
        nonlocal canvas
        if np.mean(canvas) > 250:
            result_text.set_text('Canvas is empty!\nDraw something first.')
            result_text.set_color(highlight_color)
            fig.canvas.draw_idle()
            return

        # Preprocess -> normalize -> flatten -> predict
        preprocessed = preprocess_drawn_image(canvas)
        normalized = (preprocessed.astype(np.float32) - 127.5) / 127.5
        flat_input = normalized.reshape(1, 784)

        output = model.forward(flat_input, training=False)
        predicted_digit = int(np.argmax(output[0]))
        confidence_pct = float(output[0][predicted_digit]) * 100

        # Update result display
        ax_result.clear()
        ax_result.set_facecolor(panel_color)
        ax_result.axis('off')
        
        # Determine confidence color
        if confidence_pct >= 80:
            conf_color = success_color
        elif confidence_pct >= 50:
            conf_color = '#ffa500'
        else:
            conf_color = highlight_color
        
        # Display prediction text at top of axes
        ax_result.text(0.5, 0.98, f'Predicted: {predicted_digit}', 
                      fontsize=22, fontweight='bold', color=conf_color,
                      ha='center', va='top', transform=ax_result.transAxes)
        ax_result.text(0.5, 0.82, f'Confidence: {confidence_pct:.1f}%', 
                      fontsize=12, color=text_color,
                      ha='center', va='top', transform=ax_result.transAxes)
        
        # Show preprocessed image in lower portion
        ax_result.imshow(preprocessed, cmap='gray', aspect='equal', 
                        extent=[0, 28, 0, 28])
        ax_result.set_xlim(-5, 33)
        ax_result.set_ylim(-2, 45)
        
        # Update confidence bars
        update_confidence_bars(output)
        
        fig.canvas.draw_idle()
        
        print(f"\n[PREDICTION] Digit: {predicted_digit}, "
              f"Confidence: {confidence_pct:.2f}%")

    def on_clear_clicked(event):
        nonlocal canvas, last_point, stroke_points
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        last_point = None
        stroke_points = []
        canvas_display.set_data(canvas)
        
        # Reset result area
        ax_result.clear()
        ax_result.set_facecolor(panel_color)
        # ax_result.set_title('Prediction Result', fontsize=14, color=text_color, pad=10)
        ax_result.axis('off')
        ax_result.text(0.5, 0.5, 'Draw a digit\nand click\nPredict',
                      ha='center', va='center', fontsize=18, color='#666666',
                      style='italic', transform=ax_result.transAxes)
        
        # Reset confidence bars
        for bar, label in zip(bars, bar_labels):
            bar.set_width(0)
            bar.set_color(accent_color)
            label.set_text('')
        
        fig.canvas.draw_idle()
        print("\n[INFO] Canvas cleared!")

    def on_exit_clicked(event):
        plt.close(fig)

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    btn_predict.on_clicked(on_predict_clicked)
    btn_clear.on_clicked(on_clear_clicked)
    btn_exit.on_clicked(on_exit_clicked)

    print("\n" + "=" * 50)
    print("INSTRUCTIONS:")
    print("- Hold left mouse button and drag to draw")
    print("- Click 'Predict' to get a prediction")
    print("- Click 'Clear' to start over")
    print("- Click 'Exit' or close the window to quit")
    print("=" * 50)

    plt.show()

    input("\nPress Enter to return to menu...")


# ====================================================================
#                          MAIN MENU & ENTRY
# ====================================================================

def show_menu():
    print("\n" + "=" * 50)
    print("  MNIST DIGIT RECOGNITION — NEURAL NETWORK")
    print("=" * 50)
    print("\n  1. Train Model (and save)")
    print("  2. Test Model  (load and draw)")
    print("  3. Exit")
    print("\n" + "=" * 50)


def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice (1–3): ").strip()
        if choice == '1':
            train_model()
        elif choice == '2':
            test_model()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1, 2, or 3.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()