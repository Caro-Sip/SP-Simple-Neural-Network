import numpy as np
import matplotlib.pyplot as plt

# read the ubyte file in binary mode
f = open('mnist_data/train-images-idx3-ubyte', 'rb')

# parameter of file
image_size = 28
num_images = 5

# skip the header information
f.read(16)

# read the image data in bytes into buffer (buf)
buf = f.read(image_size * image_size * num_images)

# interprets the raw bytes as an array of unsigned 8-bit pixel values (0–255)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

# arranges the flat pixel sequence into num_images images of shape 28×28 with a singleton channel dimension.
data = data.reshape(num_images, image_size, image_size, 1)

# Source - https://stackoverflow.com/a
# Posted by Punnerud, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-13, License - CC BY-SA 4.0


image_index = 0 # first image in the dataset 0 to 4 in this case
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()
