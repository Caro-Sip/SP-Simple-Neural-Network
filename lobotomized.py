import numpy as np
import nnfs
import os
import cv2
import pickle
import random

nnfs.init()


# ============== MNIST-STYLE PREPROCESSING ==============
def preprocess_drawn_image(canvas):
    """
    Preprocess a drawn image to match MNIST format:
    1. Invert colors (MNIST is white digit on black background)
    2. Find bounding box of the digit
    3. Crop and center by center of mass
    4. Resize to 20x20 (MNIST digits are ~20x20 centered in 28x28)
    5. Pad to 28x28 with digit centered
    """
    # Invert: we draw black on white, MNIST is white on black
    inverted = 255 - canvas
    
    # Threshold to get binary image
    _, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of non-zero pixels
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add small padding to bounding box
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(canvas.shape[1] - x, w + 2*pad)
    h = min(canvas.shape[0] - y, h + 2*pad)
    
    # Crop to bounding box
    cropped = inverted[y:y+h, x:x+w]
    
    # Make it square by padding the shorter side
    if h > w:
        diff = h - w
        left_pad = diff // 2
        right_pad = diff - left_pad
        cropped = cv2.copyMakeBorder(cropped, 0, 0, left_pad, right_pad, 
                                      cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        diff = w - h
        top_pad = diff // 2
        bottom_pad = diff - top_pad
        cropped = cv2.copyMakeBorder(cropped, top_pad, bottom_pad, 0, 0, 
                                      cv2.BORDER_CONSTANT, value=0)
    
    # Resize to 20x20 (MNIST digits are approximately this size within the 28x28 frame)
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Center in 28x28 image using center of mass
    # Calculate center of mass
    M = cv2.moments(resized)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 10, 10
    
    # Create 28x28 output and place the digit centered
    result = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate offset to center the center of mass at (14, 14)
    offset_x = 14 - cx - 4  # -4 because we're placing 20x20 in 28x28
    offset_y = 14 - cy - 4
    
    # Clamp offsets
    offset_x = max(0, min(8, offset_x + 4))
    offset_y = max(0, min(8, offset_y + 4))
    
    # Place the 20x20 digit in the 28x28 frame
    result[offset_y:offset_y+20, offset_x:offset_x+20] = resized
    
    return result


# ============== DATA AUGMENTATION ==============
def augment_image(img):
    """Apply random augmentations to a 28x28 grayscale image (numpy array).
    This helps the model generalize to real handwriting."""
    h, w = img.shape[:2]
    result = img.copy()
    
    # Random rotation (-15 to +15 degrees)
    if random.random() < 0.7:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random translation (shift by up to 2 pixels)
    if random.random() < 0.7:
        tx = random.uniform(-2, 2)
        ty = random.uniform(-2, 2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random scale (zoom in/out slightly)
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random shear
    if random.random() < 0.3:
        shear = random.uniform(-0.1, 0.1)
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Add Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 20), result.shape).astype(np.float32)
        result = result.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Random brightness/contrast
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # Erosion or dilation (thicken/thin strokes)
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            result = cv2.erode(result, kernel, iterations=1)
        else:
            result = cv2.dilate(result, kernel, iterations=1)
    
    return result


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


# Input layer
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


# Softmax activation
class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Adam optimizer
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# Common loss class
class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Categorical cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# Combined Softmax activation and cross-entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# Common accuracy class
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Categorical accuracy
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# Model class
class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]

                    output = self.forward(batch_X, training=False)
                    self.loss.calculate(output, batch_y)
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


# Load MNIST dataset
def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    filenames = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file),
                              cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
            filenames.append(f"{dataset}/{label}/{file}")

    return np.array(X), np.array(y).astype('uint8'), filenames


def create_data_mnist(path):
    X, y, filenames_train = load_mnist_dataset('train', path)
    X_test, y_test, filenames_test = load_mnist_dataset('test', path)
    return X, y, X_test, y_test, filenames_train, filenames_test


def save_model(model, filename='mnist_model.pkl'):
    """Save trained model weights and biases"""
    model_data = {'layers': []}
    for layer in model.trainable_layers:
        model_data['layers'].append({
            'weights': layer.weights,
            'biases': layer.biases
        })
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[OK] Model saved to '{filename}'!")


def load_model(model, filename='mnist_model.pkl'):
    """Load trained model weights and biases"""
    if not os.path.exists(filename):
        print(f"\n[ERROR] Model file '{filename}' not found!")
        print("Please train the model first (Option 1).")
        return False
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    for i, layer in enumerate(model.trainable_layers):
        layer.weights = model_data['layers'][i]['weights']
        layer.biases = model_data['layers'][i]['biases']
    print(f"\n[OK] Model loaded from '{filename}'!")
    return True


def train_model():
    """Train the neural network with data augmentation for better handwriting recognition"""
    print("\n" + "="*50)
    print("TRAINING MODE (with augmentation)")
    print("="*50)
    print("\n[INFO] Loading MNIST dataset...")
    X_images, y, X_test_images, y_test, _, _ = create_data_mnist('mnist_png')
    
    # Shuffle training data
    keys = np.array(range(X_images.shape[0]))
    np.random.shuffle(keys)
    X_images = X_images[keys]
    y = y[keys]
    
    # Prepare validation data (no augmentation)
    X_test = (X_test_images.reshape(X_test_images.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    print(f"[OK] Loaded {len(X_images)} training images and {len(X_test_images)} test images!")
    
    model = Model()
    model.add(Layer_Dense(784, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()
    
    print("\n[INFO] Starting training with augmentations...")
    print("(Each epoch uses freshly augmented training images)")
    print("-"*50)
    
    num_epochs = 10
    batch_size = 128
    
    for epoch in range(1, num_epochs + 1):
        print(f'\n=== Epoch {epoch}/{num_epochs} (augmenting data...) ===')
        
        # Apply augmentation to training images for this epoch
        X_aug = np.empty_like(X_images)
        for i in range(X_images.shape[0]):
            if random.random() < 0.8:  # 80% chance to augment
                X_aug[i] = augment_image(X_images[i])
            else:
                X_aug[i] = X_images[i]
        
        # Flatten and normalize
        X_flat = (X_aug.reshape(X_aug.shape[0], -1).astype(np.float32) - 127.5) / 127.5
        
        # Train one epoch
        model.train(X_flat, y, validation_data=(X_test, y_test),
                    epochs=1, batch_size=batch_size, print_every=100)
    
    save_model(model)
    print("\n[OK] Training complete!")
    input("\nPress Enter to return to menu...")


def test_model():
    """Test the trained model - draw your own digit with matplotlib"""
    print("\n" + "="*50)
    print("TESTING MODE - DRAW YOUR OWN DIGIT")
    print("="*50)
    
    # Try to import matplotlib
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for interactive drawing
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
    except ImportError:
        print("\n[ERROR] matplotlib is required for drawing mode.")
        print("Install it with: pip install matplotlib")
        input("\nPress Enter to return to menu...")
        return
    
    # Load model
    model = Model()
    model.add(Layer_Dense(784, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()
    
    if not load_model(model):
        input("\nPress Enter to return to menu...")
        return
    
    # Drawing canvas state
    canvas_size = 280
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    drawing = False
    last_point = None
    
    # Create figure and axes
    fig, (ax_canvas, ax_result) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Draw a digit (0-9) with your mouse', fontsize=14)
    
    # Canvas for drawing
    ax_canvas.set_title('Draw here (hold left mouse button)')
    img_display = ax_canvas.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax_canvas.axis('off')
    
    # Result display
    ax_result.set_title('Prediction will appear here')
    ax_result.axis('off')
    result_text = ax_result.text(0.5, 0.5, 'Draw a digit\nthen click\n"Predict"', 
                                  ha='center', va='center', fontsize=16,
                                  transform=ax_result.transAxes)
    
    # Add buttons
    ax_predict = plt.axes([0.3, 0.02, 0.15, 0.06])
    ax_clear = plt.axes([0.5, 0.02, 0.15, 0.06])
    ax_exit = plt.axes([0.7, 0.02, 0.15, 0.06])
    
    btn_predict = Button(ax_predict, 'Predict')
    btn_clear = Button(ax_clear, 'Clear')
    btn_exit = Button(ax_exit, 'Exit')
    
    def on_mouse_press(event):
        nonlocal drawing, last_point
        if event.inaxes == ax_canvas and event.button == 1:
            drawing = True
            last_point = (int(event.xdata), int(event.ydata))
    
    def on_mouse_release(event):
        nonlocal drawing, last_point
        drawing = False
        last_point = None
    
    def on_mouse_move(event):
        nonlocal canvas, last_point
        if drawing and event.inaxes == ax_canvas and event.xdata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                # Draw a thick line
                if last_point is not None:
                    cv2.line(canvas, last_point, (x, y), 0, 15)
                cv2.circle(canvas, (x, y), 8, 0, -1)
                last_point = (x, y)
                img_display.set_data(canvas)
                fig.canvas.draw_idle()
    
    def predict_digit(event):
        nonlocal canvas
        # Check if canvas is empty
        if np.mean(canvas) > 250:
            result_text.set_text('Canvas is empty!\nDraw something first.')
            fig.canvas.draw_idle()
            return
        
        # Preprocess using MNIST-style preprocessing
        small_image = preprocess_drawn_image(canvas)
        
        # Normalize like training data
        processed = (small_image.astype(np.float32) - 127.5) / 127.5
        processed = processed.reshape(1, 784)
        
        # Make prediction
        output = model.forward(processed, training=False)
        prediction = np.argmax(output[0])
        confidence = output[0][prediction] * 100
        
        # Get top 3 predictions
        top3_indices = np.argsort(output[0])[-3:][::-1]
        
        # Update result display
        result_str = f'Prediction: {prediction}\n'
        result_str += f'Confidence: {confidence:.1f}%\n\n'
        result_str += 'Top 3:\n'
        for i, idx in enumerate(top3_indices, 1):
            result_str += f'{i}. Digit {idx}: {output[0][idx]*100:.1f}%\n'
        
        result_text.set_text(result_str)
        
        # Show processed image (what the model actually sees)
        ax_result.clear()
        ax_result.set_title(f'Predicted: {prediction} ({confidence:.1f}%)')
        ax_result.imshow(small_image, cmap='gray')
        ax_result.axis('off')
        ax_result.text(0.5, -0.1, result_str, ha='center', va='top', 
                       fontsize=10, transform=ax_result.transAxes)
        
        fig.canvas.draw_idle()
        
        # Also print to console
        print(f"\n[PREDICTION] Digit: {prediction}, Confidence: {confidence:.2f}%")
        print(f"  Top 3: {[(idx, f'{output[0][idx]*100:.1f}%') for idx in top3_indices]}")
    
    def clear_canvas(event):
        nonlocal canvas, last_point
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        last_point = None
        img_display.set_data(canvas)
        ax_result.clear()
        ax_result.set_title('Prediction will appear here')
        ax_result.axis('off')
        result_text = ax_result.text(0.5, 0.5, 'Draw a digit\nthen click\n"Predict"', 
                                      ha='center', va='center', fontsize=16,
                                      transform=ax_result.transAxes)
        fig.canvas.draw_idle()
        print("\n[INFO] Canvas cleared!")
    
    def exit_app(event):
        plt.close(fig)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    btn_predict.on_clicked(predict_digit)
    btn_clear.on_clicked(clear_canvas)
    btn_exit.on_clicked(exit_app)
    
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("="*50)
    print("- Hold left mouse button and drag to draw")
    print("- Click 'Predict' to get prediction")
    print("- Click 'Clear' to clear canvas")
    print("- Click 'Exit' or close window to exit")
    print("="*50)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
    
    print("\n[OK] Drawing mode closed!")
    input("\nPress Enter to return to menu...")


def show_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print("MNIST DIGIT RECOGNITION - NEURAL NETWORK")
    print("="*50)
    print("\n1. Train Model (and save)")
    print("2. Test Model (load and predict)")
    print("3. Exit")
    print("\n" + "="*50)


def main():
    """Main menu loop"""
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == '1':
            train_model()
        elif choice == '2':
            test_model()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice! Please enter 1, 2, or 3.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()