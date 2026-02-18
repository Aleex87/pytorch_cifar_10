# PyTorch CIFAR-10 Deep Learning Pipeline

## Dataset and Modeling Choice

For this project, we use the CIFAR-10 dataset, which consists of 60,000 RGB images of size 32x32 pixels divided into 10 classes.

Each image has shape (32, 32, 3), meaning:
- 32 pixels in height
- 32 pixels in width
- 3 color channels (Red, Green, Blue)

This results in 32 × 32 × 3 = 3072 numerical values per image.

This project implements a complete Deep Learning pipeline using PyTorch, including:

- Data loading and preprocessing
- CNN model training
- Hyperparameter experimentation
- Model checkpointing and early stopping
- Model versioning with DVC
- Inference on new unseen images

## Environment Setup

This project uses `uv` for dependency management.

To install all required dependencies and reproduce the same environment:

```bash
uv sync
```
This command installs all dependencies defined in the project configuration and ensures reproducibility.

## Training the Model

The entire training pipeline can be executed using:
python main.py
This will:

1. Load the dataset
2. Build the CNN model
3. Train the network
4. Monitor validation loss
5. Apply early stopping
6. Automatically save only the best model inside the models/ directory

All trained models are tracked using DVC to avoid committing large binary files to Git.

# Running Inference (Testing the Model)

To test the trained model on new images:
Create a folder in the project root named:
> inference_images/

Drag and drop one or more .png, .jpg, or .jpeg images into this folder.

Run:
python -m src.inference

The script will:

- Load the best trained model
- Apply the same preprocessing used during training
- Predict the class for each image
- Print the predicted class and confidence score

## Supported Classes

The model was trained on CIFAR-10 and can classify the following 10 categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

If an image outside these categories is provided, the model will still output one of these 10 classes, since it was trained in a closed-set classification setting.


## Custom Data Preparation Pipeline

The dataset was manually downloaded in its original CIFAR-10 binary format (cifar-10-batches-py).

The raw dataset consists of serialized batch files where each image is stored as a flattened vector of 3072 values (32 × 32 × 3).

Instead of relying on prebuilt dataset loaders, a dedicated preprocessing script (prepare_data.py) was implemented.

This script performs the following operations:

1. Reads CIFAR-10 batch files using Python's pickle
2. Extracts image data and corresponding labels
3. Reshapes each flattened image from 3072 values into a 32×32×3 tensor
4. Converts NumPy arrays into PNG image files
5. Saves images into class-specific directories

The final dataset structure follows a standard image classification layout:

data/processed/
    train/
        airplane/
        automobile/
        ...
    test/
        airplane/
        automobile/
        ...

This approach ensures:

- Full transparency in data handling
- Explicit transformation of raw numerical arrays into image format
- Compatibility with PyTorch dataset utilities

> **Note:**
**The raw dataset is not included in the repository.**
**It must be downloaded manually and placed inside `data/raw`.**
https://www.kaggle.com/datasets/harshajakkam/cifar-10-python-cifar-10-python-tar-gz


### Reproducibility of the preprocessing pipeline

By implementing a custom preprocessing stage, the project demonstrates an understanding of how raw image datasets are internally structured and how they can be programmatically reconstructed into usable formats.



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
│   ├── prepare_data.py
│   └── inference.py
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

> inference.py  
Loads a trained model and performs predictions on new unseen images.

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
> data/raw

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

The workflow is reproducible thanks to:

- Version-controlled dependencies (uv.lock)
- DVC-managed datasets and models
- Modular architecture
- Controlled experiment configurations

## Training Procedure and Model Checkpointing

After verifying that the data pipeline and CNN architecture were correctly implemented, the training process was executed using `main.py`.

During the first run, we monitored the training and validation loss values printed in the terminal. We observed that while the training loss continuously decreased, the validation loss reached a minimum at a certain epoch and then started to increase slightly. This behavior indicates the beginning of overfitting.

Terminal output example:
```bash
using device: cpu

Epoch 1/5
---------------------------------------------
Train Loss: 1.2733
Validation Loss: 1.0176

Epoch 2/5
---------------------------------------------
Train Loss: 0.9078
Validation Loss: 0.8769

Epoch 3/5
---------------------------------------------
Train Loss: 0.7585
Validation Loss: 0.8973

Epoch 4/5
---------------------------------------------
Train Loss: 0.6402
Validation Loss: 0.8890

Epoch 5/5
---------------------------------------------
Train Loss: 0.5378
Validation Loss: 0.8966
```

To address this, we introduced two important mechanisms:

#### 1. Model Checkpointing

The model is saved whenever the validation loss improves. This ensures that we keep the best-performing version of the model instead of only the final epoch.

```python
torch.save(model.state_dict(), "models/best_model.pth")
```
This file contains the learned parameters (weights) of the network and can later be loaded for inference or evaluation.

### 2. Early Stopping

A simple early stopping mechanism was implemented using a patience counter. If the validation loss does not improve for a predefined number of epochs, the training process is stopped early.
This prevents unnecessary overfitting and reduces training time.

As a result, the training process now:
- Monitors validation loss
- Saves the best model automatically
- Stops training when no further improvement is observed

The trained models are tracked using DVC to ensure proper version control of large binary artifacts without committing them directly to Git.

## Experiments

To evaluate the effect of different hyperparameters, three experiments were conducted.

Each experiment was trained using early stopping and model checkpointing. The best model (based on validation loss) was automatically saved.

### Experiment 1 – Baseline

- Learning rate: 0.001
- Batch size: 32
- Epochs: 5

### Experiment 2 – Lower Learning Rate

- Learning rate: 0.0005
- Batch size: 32
- Epochs: 5

The hypothesis was that a lower learning rate would result in a slower but more stable convergence.

### Experiment 3 – Larger Batch Size

- Learning rate: 0.001
- Batch size: 64
- Epochs: 5

The hypothesis was that a larger batch size would produce more stable gradient updates.

---

### Results Summary

| Experiment | Learning Rate | Batch Size | Best Epoch | Best Validation Loss |
|------------|--------------|------------|------------|----------------------|
| Exp 1      | 0.001        | 32         | 4          | 0.8277               |
| Exp 2      | 0.0005       | 32         | 5          | 0.8445               |
| Exp 3      | 0.001        | 64         | 5          | 0.8361               |

The best performing configuration will be selected based on the lowest validation loss.

## Model Comparison and Analysis

### Best Performing Configuration

The best performing configuration was:

- **Learning Rate:** 0.001  
- **Batch Size:** 32  
- **Best Validation Loss:** 0.8277  

This configuration achieved the lowest validation loss among the three experiments.

### Learning Rate Comparison (0.001 vs 0.0005)

Reducing the learning rate from 0.001 to 0.0005 resulted in:

- A smoother and more gradual decrease in validation loss
- Slower convergence
- Slightly worse performance within 5 epochs

Although the lower learning rate produced stable training behavior, it likely requires more epochs to reach competitive performance. With only 5 epochs, it did not outperform the baseline configuration.

### Batch Size Comparison (32 vs 64)

Increasing the batch size from 32 to 64 resulted in:

- More stable gradients
- Reduced variance during training
- No clear signs of overfitting

However, the final validation performance was slightly inferior to the batch size of 32.
This can be explained by the fact that with the same number of epochs:

- A smaller batch size performs more parameter updates per epoch
- A larger batch size performs fewer updates per epoch

Therefore, within a limited training duration (5 epochs), batch size 32 achieved slightly better generalization.

### Final Observation

For the current architecture and training setup, a learning rate of 0.001 and batch size of 32 provide the best trade-off between convergence speed and validation performance.





