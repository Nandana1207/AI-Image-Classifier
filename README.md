AI Image Classifier

This project is an AI-powered image classifier trained using Teachable Machine. It uses TensorFlow and OpenCV to classify images into different categories (e.g., Cat or Dog).

FEATURES

Loads a trained AI model (.h5 format).
Accepts image input and processes it for classification.
Predicts the class of the given image with confidence scores.

REQUIREMENTS

Before running the code, install the following dependencies:

pip install tensorflow numpy opencv-python

How to Use?

1. Clone the Repository

git clone https://github.com/Nandana1207/AI-Image-Classifier.git
cd AI-Image-Classifier

2. Upload the Model and Image Files

Place the trained model file (keras_model.h5) in the project folder.

Upload the image you want to classify (test.jpg).


3. Run the Code

Use the following script to classify an image:

import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("keras_model.h5")

# Load image
image_path = "test.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))  # Resize to match model input
image = np.array(image) / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(image)

# Define class labels
class_labels = ["Cat", "Dog"]  # Modify as needed
predicted_class = np.argmax(prediction)

print("Predicted Class:", class_labels[predicted_class])

Example OUTPUT

Prediction: [[0.90441644 0.0955836]]
Predicted Class: Cat

CONTRINUTING

Feel free to contribute by improving the model, adding more categories, or optimizing the code.

LICENSE

This project is open-source and free to use.
