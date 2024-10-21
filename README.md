# Project Description

This project aims to predict image displacement and strain using a Swin Transformer network. It implements a deep learning model based on Swin Transformer that can extract displacement and strain information from deformed images and reference images. The model is trained and tested on synthetic datasets that include various types of deformation modes.

## Main Features

- **Dataset Generation**: Use the `dataset.py` script to generate synthetic deformed image datasets, supporting multiple deformation modes.
- **Model Training**: Utilize `strain_train.py` and `displacement_train.py` scripts for model training, with support for custom data augmentation and learning rate scheduling.
- **Result Prediction**: Use `strain_test.py` and `displacement_test.py` scripts for model prediction, saving results as CSV files.
- **Result Visualization**: Generate displacement cloud maps using the `disp_cloudmap.py` script, with support for overlay display on original images.

## Pre-trained Parameters

Pre-trained model parameters and tesile test images are saved on Google Drive. You can download them via the following links:
- [Pre-trained model parameters and tesile test images](https://drive.google.com/drive/folders/1cYueAAM_ONtNQbNL_yVudDYnKMm1jG34?usp=drive_link)


## Usage Instructions

1. Download and extract pre-trained model parameters and experiment images.
2. Use `strain_train.py` or `displacement_train.py` for model training.
3. Use `strain_test.py` or `displacement_test.py` for model prediction.
4. Use `disp_cloudmap.py` for result visualization.

## The dataset
The complete dataset will be uploaded after the paper is published.

For any questions, please refer to the code comments in the project or contact the project maintainer.
