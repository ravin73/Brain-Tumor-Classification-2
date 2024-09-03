# Brain Tumor Classification Using CNN

This project is a machine learning-based approach to classify brain tumors from MRI images. The model is built using a Convolutional Neural Network (CNN) and is deployed as a web application using Flask. The trained model is also converted to TensorFlow Lite for deployment on mobile or edge devices.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Conversion](#model-conversion)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Early detection of brain tumors is critical for effective treatment. This project aims to assist in the early diagnosis of brain tumors using machine learning techniques. The model can classify MRI images as either "No Brain Tumor" or "Yes Brain Tumor".

## Dataset
The dataset used for this project consists of labeled MRI images. The images are divided into two categories:
- **No Tumor:** Images that do not show signs of a brain tumor.
- **Yes Tumor:** Images that indicate the presence of a brain tumor.

The images are preprocessed to a size of 64x64 pixels and normalized before being fed into the CNN model.

## Project Structure

## Model Architecture
The CNN model consists of the following layers:
- **Conv2D**: 3 layers with 32, 32, and 64 filters, each followed by ReLU activation and MaxPooling.
- **Flatten**: Converts the 2D matrix to a 1D vector.
- **Dense**: 64 neurons with ReLU activation.
- **Dropout**: 0.5 rate to prevent overfitting.
- **Dense**: Output layer with 2 neurons and softmax activation for binary classification.

The model is trained using the categorical cross-entropy loss function and the Adam optimizer.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
2. Installed the required dependencies
   ~pip install -r requirements.txt
3. Run the flask application
   ~python app.py

