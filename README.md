
# CNN Drift Detection and Model Retraining

This repository contains code for a Convolutional Neural Network (CNN) used for detecting data drift and retraining the model accordingly. The CNN is trained on the MNIST dataset and is capable of detecting drift by monitoring the performance of the model on a ground truth dataset.

## Introduction

The code in this repository is designed to demonstrate the process of detecting data drift in a machine learning model and retraining the model when drift is detected. It achieves this by generating rotated samples from the MNIST dataset and monitoring the performance of the model on a separate ground truth dataset.

## Installation

To run the code, you'll need Python 3.x and the following libraries:

- NumPy
- Matplotlib
- Keras
- PyTorch
- Scikit-learn

You can install these dependencies using pip:

```
pip install numpy matplotlib keras torch scikit-learn
```

## Results

The code will output the training loss over epochs and the monitoring accuracy. If drift is detected, it will print a message indicating so.