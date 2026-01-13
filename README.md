# SP-Simple-Neural-Network
SP semester project

https://pypi.org/project/mnist-datasets/

`train` files are for model training—contain
train files are for model training—contain 60,000 samples; these are used to fit parameters.

`t10k` (“test10k”) files are for evaluation—contain 10,000 samples; these are held out and not used while training, only for final performance measurement.

Both have the same format (images and labels), but content is different; train = learning, t10k = unbiased accuracy check.

First output (10, 1) array:

Raw sigmoid activations from each output neuron
Each value is independent: sigmoid(z) = 1/(1+e^(-z))
Values are in (0,1) but don't sum to 1 (sum ≈ 3.5)
Not a probability distribution
Each neuron independently fires between 0 and 1
Second output (0-9 list):

Softmax probabilities computed from logits
Values form a proper probability distribution: sum = 1.0
Represents the network's confidence that the input is each digit
softmax(z_i) = e^(z_i) / Σe^(z_j)
Class 3 has 68% probability, class 6 has 25%, etc.