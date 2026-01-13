import struct
import numpy as np
import matplotlib.pyplot as plt
from main import feedforward, sigmoid


def read_first_image(image_path, label_path=None):
	with open(image_path, 'rb') as f:
		magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
		buf = f.read(rows * cols)  # read first image only
		img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		img = img.reshape(rows, cols)

	label = None
	if label_path is not None:
		with open(label_path, 'rb') as lf:
			lm, ln = struct.unpack('>II', lf.read(8))
			lb = lf.read(1)
			if lb:
				label = int.from_bytes(lb, 'big')

	return img, label


if __name__ == '__main__':
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

	# simple network: 784 -> 64 -> 16 -> 1
	np.random.seed(50)
	input_size = x.size
	sizes = [input_size, 64, 16, 10]
	Ws = [np.random.randn(sizes[i+1], sizes[i]).astype(np.float32) for i in range(len(sizes)-1)]
	bs = [np.random.randn(sizes[i+1], 1).astype(np.float32) for i in range(len(sizes)-1)]

	# prepare input vector and compute hidden activations using feedforward
	x_vec = x.reshape(-1, 1).astype(np.float32)

	# use `feedforward` for all layers except the final one so we can
	# compute final logits here and apply softmax (keeps this change
	# local to this file, as requested)
	if len(Ws) > 1:
		x_vec = feedforward(Ws[:-1], bs[:-1], x_vec)

	# final layer (logits) -- linear, no activation
	W_last, b_last = Ws[-1], bs[-1]
	logits = (W_last.dot(x_vec) + b_last).astype(np.float64).ravel()

	# softmax -> probabilities in [0,1] summing to 1
	def softmax(z):
		z = z - np.max(z)
		exp = np.exp(z)
		return exp / exp.sum()

	probs = softmax(logits)
	pred = int(probs.argmax())

	# print each class probability on its own line (0..9)
	for i, p in enumerate(probs):
		print(f"{i}: {p:.6f}")
	print('predicted class =', pred)
	if label is not None:
		print('true label =', label)