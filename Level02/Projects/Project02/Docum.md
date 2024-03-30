# Chest X-Ray Image Classification

This notebook outlines the development of a deep learning classifier for chest X-ray images. It aims to distinguish between different conditions based on X-ray images by employing convolutional neural networks (CNNs).

## Setup

We begin by importing the necessary Python libraries for data handling, image processing, visualization, and deep learning.

```python
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from vit_keras import vit
from sklearn.metrics import classification_report, confusion_matrix
```

## Preprocessing

The `ImagePreprocessor` class is responsible for preparing images for the neural network. This includes resizing the images to a consistent shape and normalizing pixel values.

### ImagePreprocessor Class

```
class ImagePreprocessor:
    # Initialization method
    def init(self, target_size=(224, 224)):
        self.target_size = target_size    # Method to resize and normalize images
    def resize_and_normalize_image(self, image):
        # Resize image
        image = cv2.resize(image, self.target_size)
        # Convert color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        return image    # Public method to preprocess images
    def preprocess_image(self, image):
        return self.resize_and_normalize_image(image)
```

## ChestXRayClassifier

The `ChestXRayClassifier` class encompasses the workflow for building, training, and evaluating the models.

### Class Initialization

Upon instantiation, the class initializes the data directory, input dimensions, batch size, and image preprocessor. It also configures the data augmentation pipeline using `ImageDataGenerator`.

### Sample Image Display

The `display_sample_images` method visualizes a subset of images from each class.

### Class Distribution Plotting

The `plot_class_distribution` method visualizes the distribution of classes in the dataset.

## Model Building

We explore two architectures: Vision Transformer (ViT) and EfficientNet, implemented in `build_vit_model` and `build_efficientnet_model` methods, respectively.

### Vision Transformer (ViT)

The Vision Transformer applies the transformer architecture to image classification tasks.

### EfficientNet

EfficientNet scales convolutional networks and achieves state-of-the-art results.

## Model Compilation and Training

The `compile_and_train` method compiles the chosen model with an appropriate loss function and optimizer. It then proceeds to train the model using the augmented image data.

## Model Evaluation

The `evaluate_model` method evaluates the trained model's performance on a test set and provides a confusion matrix and classification report for detailed analysis.

## Visualization

The `plot_history` method plots the training history, showing accuracy and loss over epochs for both training and validation sets.
