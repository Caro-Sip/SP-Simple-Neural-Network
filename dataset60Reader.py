import struct
import numpy as np
import matplotlib.pyplot as plt
from main import feedforward, sigmoid


def read_mnist_image(image_path, label_path=None, index=0):
	"""
	Read a single MNIST image and its label from ubyte files.
	
	Args:
		image_path: Path to the images idx3-ubyte file
		label_path: Path to the labels idx1-ubyte file (optional)
		index: Index of the image to read (default: 0)
	
	Returns:
		tuple: (image, label) where image is (28, 28) numpy array, 
		       label is int or None if label_path not provided
	"""
	with open(image_path, 'rb') as f:
		magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
		# skip to the desired image
		f.seek(16 + index * rows * cols)
		buf = f.read(rows * cols)
		img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		img = img.reshape(rows, cols)

	label = None
	if label_path is not None:
		with open(label_path, 'rb') as lf:
			lm, ln = struct.unpack('>II', lf.read(8))
			lf.seek(8 + index)
			lb = lf.read(1)
			if lb:
				label = int.from_bytes(lb, 'big')

	return img, label


def softmax(z):
	"""Compute softmax probabilities from logits."""
	z = z - np.max(z)
	exp = np.exp(z)
	return exp / exp.sum()


def predict_digit(img, Ws, bs, return_probs=False):
	"""
	Run inference on a single MNIST image through a neural network.
	
	Args:
		img: (28, 28) image array or flattened (784,) array
		Ws: List of weight matrices for each layer
		bs: List of bias vectors for each layer
		return_probs: If True, return (predicted_class, probabilities),
		              otherwise just return predicted_class
	
	Returns:
		int or tuple: Predicted class (0-9) or (predicted_class, probs)
	"""
	# normalize and flatten image
	if img.ndim == 2:
		x = (img.flatten() / 255.0).astype(np.float32)
	else:
		x = (img / 255.0).astype(np.float32)
	
	x_vec = x.reshape(-1, 1).astype(np.float32)

	# use feedforward for all layers except the final one
	if len(Ws) > 1:
		x_vec = feedforward(Ws[:-1], bs[:-1], x_vec)

	# final layer (logits) -- linear, no activation
	W_last, b_last = Ws[-1], bs[-1]
	logits = (W_last.dot(x_vec) + b_last).astype(np.float64).ravel()

	# softmax -> probabilities in [0,1] summing to 1
	probs = softmax(logits)
	pred = int(probs.argmax())

	if return_probs:
		return pred, probs
	return pred


if __name__ == '__main__':
	# Example usage: read first image and predict
	img, label = read_mnist_image('mnist_data/train-images-idx3-ubyte',
								  'mnist_data/train-labels-idx1-ubyte',
								  index=0)

	print('image shape =', img.shape)

	# optional display
	try:
		plt.imshow(img, cmap='gray', block=True)
		plt.title(f'Training image (label={label})')
		plt.axis('off')
		plt.show()
	except Exception:
		pass

	# create a simple network: 784 -> 64 -> 16 -> 10
	np.random.seed(50)
	input_size = 784
	sizes = [input_size, 64, 16, 10]
	Ws = [np.random.randn(sizes[i+1], sizes[i]).astype(np.float32) for i in range(len(sizes)-1)]
	bs = [np.random.randn(sizes[i+1], 1).astype(np.float32) for i in range(len(sizes)-1)]

	# run prediction
	pred, probs = predict_digit(img, Ws, bs, return_probs=True)

	# print each class probability on its own line (0..9)
	for i, p in enumerate(probs):
		print(f"{i}: {p:.6f}")
	print('predicted class =', pred)
	if label is not None:
		print('true label =', label)