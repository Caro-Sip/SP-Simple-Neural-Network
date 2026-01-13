from mnist_datasets import MNISTLoader

# This python program downloads the MNIST dataset and loads it using the MNISTLoader class
# It saves as './mnist_data/' folder by default
# IMPORTANT: you need to run 'pip install mnist-datasets' to install the required package before running this code

loader = MNISTLoader(folder='./mnist_data/')
images, labels = loader.load()
assert len(images) == 60000 and len(labels) == 60000

# Load test dataset
test_images, test_labels = loader.load(train=False)
assert len(test_images) == 10000 and len(test_labels) == 10000