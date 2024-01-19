# data_preprocessing.py

import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# path_to_dataset=https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
# Configuration
dataset_path = 'path_to_dataset'  # Replace with actual path
patch_size = 50  # size of the image patches (50x50 pixels)

# Normalize pixel values
def normalize_pixels(image):
    return tf.cast(image, tf.float32) / 255.0

# Load images and labels
def load_data(path):
    images = []
    labels = []
    
    # Iterate through the dataset folder
    for label_type in ['0', '1']:  # '0' for negative, '1' for positive
        label_folder = os.path.join(path, label_type)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (patch_size, patch_size))
            image = normalize_pixels(image)

            images.append(image)
            labels.append(int(label_type))

    return np.array(images), np.array(labels)

# Data Augmentation
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load and split the dataset
images, labels = load_data(dataset_path)
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)
