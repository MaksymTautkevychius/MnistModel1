# MnistM1 Deep Learn Model (98.50% accuracy on  Mnist dataset 10 labels)
### The model consists of two identical convolutional layers followed by a fully connected classification layer.


## Convolutional Layer 1: Conv2d() -> ReLU() -> MaxPooling()

Conv2d(in_channels=1, out_channels=hidden_units, kernel_size=3, stride=1, padding=1) → Applies a 3×3 convolution to extract low-level image features (edges, textures).
ReLU() → Introduces non-linearity, allowing the model to learn complex patterns.
MaxPool2d(kernel_size=2, stride=2) → Reduces spatial dimensions by half (28×28 → 14×14), improving computational efficiency.

## Convolutional Layer 2: Conv2d() -> ReLU() -> MaxPooling() 

Another Conv2d() → Further refines feature extraction.
ReLU() → Activation function to introduce non-linearity.
MaxPool2d(kernel_size=2, stride=2) → Again reduces dimensions (14×14 → 7×7) for better feature representation.

## Classification layer: Flatten() ->Linear()

Flatten() → Converts the feature map (batch_size, hidden_units, 7, 7) into a 1D vector for the linear layer.
Linear(hidden_units × 7 × 7 → 10) → Fully connected layer that outputs 10 class scores, one for each digit (0-9).

Loss function -> Cross Entropy Loss function <-->
Optimizer -> Sochastic gradient descent

![image](https://github.com/user-attachments/assets/a244be96-d6ec-427b-bebd-900eeb0e5b58)
