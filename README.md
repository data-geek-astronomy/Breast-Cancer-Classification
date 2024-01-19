

# Breast Cancer Classification with Deep Learning

This repository contains the implementation of a deep learning model for the classification of breast cancer using histopathological images. The project leverages TensorFlow and Keras to build and train a Convolutional Neural Network (CNN) on a dataset containing benign and malignant breast tissue samples. Transfer learning with the pre-trained VGG16 model is employed to enhance the model's performance.

## Project Overview

- **Objective**: Develop an accurate and efficient model for breast cancer classification.
- **Technologies Used**: TensorFlow, Keras, Scikit-learn.
- **Key Features**:
  - Image normalization and augmentation for data quality and consistency.
  - CNN with transfer learning (VGG16 model) for capturing spatial hierarchies in image data.
  - Evaluation metrics focused on minimizing false negatives and false positives.

## Repository Structure

- `data_preprocessing.py`: Script for loading, normalizing, and augmenting the dataset.
- `cancernet_model.py`: Definition of the CNN model using VGG16 for transfer learning.
- `train_cancernet.py`: Script for training the model.
- `evaluate_model.py`: Evaluation of the model using metrics like confusion matrix and recall.
- `requirements.txt`: List of Python packages required to run the code.
- `README.md`: This file, describing the project and how to run the code.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. The code is tested on Python 3.8.

### Installation

1. Clone this repository:
