# Cat vs. Dog Image Classification using SVM and VGG16

## Introduction

This project demonstrates a cat vs. dog image classifier using a Support Vector Machine (SVM) and feature extraction from a pre-trained VGG16 model. The dataset is sourced from Kaggle.

## Methodology

The classification process involves these steps:

1. **Dataset Acquisition:**
   - The dataset is downloaded from Kaggle (you'll need to specify the exact dataset name or ID in the code). The code handles downloading and extracting the data.

2. **Data Organization:**
    - The downloaded data is organized into 'cats' and 'dogs' subfolders for easier processing.

3. **Feature Extraction using VGG16:**
   - A pre-trained VGG16 model (without its classification head) extracts features from the images.
   - Images are resized to 64x64 before being passed through the VGG16 convolutional layers.
   - The output feature maps are flattened into a 1D vector.

4. **Data Preprocessing:**
   - Image pixel values are normalized (scaled between 0 and 1).

5. **SVM Training:**
   - An SVM classifier (using `sklearn.svm.SVC`) is trained on the extracted features and corresponding labels (cat or dog).
   - The Radial Basis Function (RBF) kernel is used (though other kernels can be explored).

6. **Prediction & Evaluation:**
   - The trained SVM model predicts the class (cat or dog) for new images.
   - The model's performance is evaluated using metrics like accuracy, classification report, and confusion matrix.

7. **Visualization:**
   - Sample predictions are visualized along with the original images and true labels.

## Support Vector Machine (SVM) Overview

SVM is a supervised learning algorithm that finds the optimal hyperplane to separate data points of different classes. Key concepts include:

- **Hyperplane:** The decision boundary that separates classes.
- **Margin:** The distance between the hyperplane and the closest data points (support vectors).  A larger margin is generally better.
- **Support Vectors:** The data points closest to the hyperplane, influencing its position.
- **Kernels:** Functions that map data to higher dimensions to handle non-linearly separable cases (e.g., RBF kernel).
- **Regularization (C parameter):** Controls the trade-off between maximizing the margin and minimizing misclassifications.

## Running the Project

Install the required libraries:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn joblib kaggle

 Clone the Repository:
   ```bash
   git clone (https://github.com/PrayashMishra022/PRODIGY-ML-03))  
   cd cat-dog-svm
