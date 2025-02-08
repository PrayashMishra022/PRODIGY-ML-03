#Code for the MODEL

import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the trained model
import random
import matplotlib.pyplot as plt

# Upgrade KaggleHub to the latest version
os.system("pip install --upgrade kagglehub")
import kagglehub

# Download dataset using KaggleHub
path = kagglehub.dataset_download("tongpython/cat-and-dog")
print("Path to dataset files:", path)

# Define paths for training and testing datasets
train_path = os.path.join(path, 'training_set', 'training_set')
test_path = os.path.join(path, 'test_set', 'test_set')
categories = [folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))]
print("Detected categories in training set:", categories)

# Load pre-trained VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x = Flatten()(base_model.output)  # Optimize feature extraction
model = Model(inputs=base_model.input, outputs=x)

@tf.function(reduce_retracing=True)
def extract_features(img):
    return model(img, training=False)

# Load and preprocess training images
X, y = [], []
for label, category in enumerate(categories):
    folder_path = os.path.join(train_path, category)
    for file in os.listdir(folder_path)[:1000]:  # Use more images for better training
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            img = np.expand_dims(img, axis=0)
            features = extract_features(img).numpy().flatten()
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model
svm = SVC(kernel='rbf', probability=True)  # Using RBF kernel and probability estimation
svm.fit(X_scaled, y)

# Save trained SVM model and scaler
joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("SVM model and scaler saved successfully.")

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    return np.expand_dims(img, axis=0)

# Function to test random images
def predict_random_image():
    category = random.choice(categories)
    test_folder = os.path.join(test_path, category)
    test_image_file = random.choice(os.listdir(test_folder))
    test_image_path = os.path.join(test_folder, test_image_file)

    test_img = preprocess_image(test_image_path)
    test_features = extract_features(test_img).numpy().flatten().reshape(1, -1)
    test_features = scaler.transform(test_features)

    prediction_prob = svm.predict_proba(test_features)[0]
    predicted_label = categories[np.argmax(prediction_prob)]
    confidence = np.max(prediction_prob)
    
    print(f"Actual: {category}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")
    plt.imshow(cv2.imread(test_image_path)[:, :, ::-1])
    plt.title(f"Actual: {category}, Predicted: {predicted_label}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Loop to ask user if they want to test a random image
while True:
    user_input = input("Do you want to test a random image from the dataset? (yes/no): ").strip().lower()
    if user_input == 'yes':
        predict_random_image()
    elif user_input == 'no':
        print("Exiting... Goodbye!")
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
