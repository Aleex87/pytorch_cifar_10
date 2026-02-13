# pytorch_cifar_10

## Dataset and Modeling Choice

For this project, we use the CIFAR-10 dataset, which consists of 60,000 RGB images of size 32x32 pixels divided into 10 classes.

Each image has shape (32, 32, 3), meaning:
- 32 pixels in height
- 32 pixels in width
- 3 color channels (Red, Green, Blue)

This results in 32 × 32 × 3 = 3072 numerical values per image.

### Why not use a fully connected network (MLP)?

In class, we have seen that images can be flattened into a vector of 3072 features and then passed through a fully connected neural network (MLP). 

In that case, the model would look like:

Input (3072 features) → Hidden Layers → Output (10 classes)

While this approach works, it has an important limitation:
when we flatten the image, we lose the spatial structure of the data. The model no longer knows which pixels are close to each other.

### Why use a Convolutional Neural Network (CNN)?

In this project, we use a Convolutional Neural Network (CNN).

Unlike an MLP, a CNN does not flatten the image at the beginning. Instead, it processes the image as a 3×32×32 tensor and applies convolutional filters that detect local patterns such as edges, shapes, and textures.

This makes CNNs more suitable for image classification tasks because:

- They preserve spatial information
- They use fewer parameters compared to fully connected layers
- They learn hierarchical features (from simple edges to complex objects)

For these reasons, even though flattening the image is simpler and was used in class examples, a CNN is a more realistic and appropriate choice for this project.

### EDA and Data Verification

Even though CIFAR-10 is a well-known and balanced dataset, we performed a brief Exploratory Data Analysis (EDA) to:

- Verify dataset integrity
- Confirm class balance
- Inspect image dimensions
- Compute mean and standard deviation for normalization

This ensures that the dataset is correctly prepared and ready for training in a reproducible pipeline.
