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

## Project Structure and Software Architecture

To ensure modularity, readability, and reproducibility, the project is structured following a separation-of-concerns design. Each component of the machine learning pipeline is isolated into dedicated modules.
The directory structure is organized as follows:

k2-ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── config.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md

## Modular Design

The pipeline is divided into logically separated modules:

> dataset.py
Responsible for loading the dataset, applying transformations, and creating DataLoaders.

> model.py
Contains the CNN architecture implemented as a subclass of torch.nn.Module.

> train.py
Implements the training loop, optimizer step, and loss computation.

> evaluate.py
Handles model evaluation and metric computation.

> utils.py
Contains helper functions such as seed setting and metric saving.

> config.py
Centralizes hyperparameters and experiment configuration.

> main.py
Acts as the entry point of the project.
It orchestrates the entire workflow:
dataset loading → model initialization → training → evaluation → result saving.

This modular approach improves readability, maintainability, and reproducibility.

## Virtual Environment and Dependency Management

The project uses uv for dependency and environment management.
Instead of using pip, dependencies were added using:
> uv add torch torchvision matplotlib pandas numpy scikit-learn jupyter dvc

This approach ensures:
- Reproducible dependency resolution
- Automatic lock file generation (uv.lock)
- Clean and isolated virtual environment (.venv)

The files:
- pyproject.toml
- uv.lock

are version controlled to guarantee reproducibility of the software environment.
The .venv directory is excluded from version control.

# Data Versioning with DVC

To ensure reproducibility and proper handling of large files, we use Data Version Control (DVC).
Since image datasets are large and unsuitable for Git tracking, DVC is used to:

- Track dataset versions
- Avoid uploading raw images to GitHub
- Maintain reproducibility of experiments

The dataset is stored in:
> data/row

and tracked using:
> dvc add data/raw

This creates a .dvc file that is tracked by Git, while the raw data itself is not uploaded to the repository.
This separation ensures:

- Clean repository
- Efficient version control
- Professional ML pipeline management.

## Reproducibility

The entire workflow can be executed using:
> python main.py

All components are deterministic and reproducible thanks to:

- Fixed random seeds
- Centralized configuration
- Version-controlled dependencies
- DVC-managed datasets

This guarantees that experiments can be reproduced consistently across different environments.

## Dataset Acquisition and Preprocessing Strategy

For this project, the dataset was not loaded using prebuilt utilities such as sklearn.datasets or torchvision.datasets.CIFAR10.
Instead, the dataset was manually downloaded from Kaggle in raw format. The original data was provided as NumPy arrays (.npy) and corresponding label files.
To create a structured and reusable dataset format, a custom preprocessing script was implemented. This script:

- Loaded the raw NumPy arrays
- Mapped each image to its corresponding label
- Converted array data into image format
- Saved images into class-specific directories
- The final dataset structure follows the standard image classification layout:

data/raw/
    train/
        airplane/
        automobile/
        ...
    test/
        airplane/
        automobile/
        ...


This approach ensures:

- Full control over the preprocessing pipeline
- Transparency in data handling
- Reproducibility of dataset preparation
- Compatibility with PyTorch custom Dataset classes

By explicitly handling the dataset conversion process, the project avoids relying on prebuilt dataset loaders and demonstrates a deeper understanding of data preparation workflows.