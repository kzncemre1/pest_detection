
# Pest Detection Mobile Application

This project is a mobile application for pest detection, developed with Flutter, PyTorch, and Flask.
Deep learning models were trained using PyTorch and deployed via a Flask-based REST API.

The Flutter mobile interface communicates with the server to send images and display classification results to the user.

## Model Training Scripts
The Python scripts for training the models are located in the `model trainings` directory. These files can be executed to train the pest detection models and save the resulting pre-trained weights and artifacts.

## Models Used

-ConvNeXt-Small

-ConvNeXt-Base

-Swin-Small

-ResNetV2-101x1-Bitm

## Dataset

The pest detection models were trained on the [IP102 Dataset] available on Kaggle.
