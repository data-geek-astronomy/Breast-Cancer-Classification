# Breast Cancer Classification with Deep Learning

Welcome to the repository for breast cancer classification using deep learning techniques! Here, we've developed a robust model leveraging state-of-the-art technologies like TensorFlow and Keras. Our focus is on accurately identifying benign and malignant breast tissue samples from histopathological images. By employing transfer learning with the VGG16 model, we enhance the model's ability to extract relevant features from the images, thus improving classification accuracy.

## Project Overview

- **Objective**: Our goal is to create an efficient and accurate model for breast cancer classification, crucial for early diagnosis and treatment.
- **Technologies Used**: TensorFlow, Keras, Scikit-learn.
- **Key Features**:
  - Data preprocessing: We ensure data quality and consistency through image normalization and augmentation.
  - Transfer learning: By utilizing the pre-trained VGG16 model, we capture spatial hierarchies in the image data, enhancing the model's performance.
  - Evaluation metrics: We focus on minimizing false negatives and false positives, crucial for medical diagnosis.

## Repository Structure

- `data_preprocessing.py`: This script handles loading, normalizing, and augmenting the dataset to prepare it for model training.
- `cancernet_model.py`: Here, we define the architecture of our CNN model, integrating the VGG16 model for transfer learning.
- `train_cancernet.py`: Use this script to train the model on your dataset.
- `evaluate_model.py`: After training, evaluate the model's performance using metrics like confusion matrix and recall.
- `requirements.txt`: This file lists all the Python packages required to run the code, ensuring smooth setup.
- `README.md`: You're currently reading it! This file provides an overview of the project and instructions for running the code.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. The code is tested on Python 3.8.

### Installation

1. Clone this repository to your local machine.
2. Install the required packages listed in `requirements.txt` using pip:

```bash
pip install -r requirements.txt

License
This project is licensed under the MIT License.

Acknowledgments
The dataset used for this project focuses on Invasive Ductal Carcinoma (IDC), the most common form of breast cancer.
We extend our gratitude to the TensorFlow and Keras communities for their invaluable tools and resources that made this project possible.
Feel free to customize and enhance this repository further for your specific needs. Happy classifying!
