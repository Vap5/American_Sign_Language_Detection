import streamlit as sl
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the categories
categories = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 16: "g", 17: "h", 18: "i",
              19: "j", 20: "k", 21: "l", 22: "m", 23: "n", 24: "o", 25: "p", 26: "q", 27: "r",
              28: "s", 29: "t", 30: "u", 31: "v", 32: "w", 33: "x", 34: "y", 35: "z"}

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.Grayscale(),  # Ensure image is in grayscale
    transforms.ToTensor()
])

# Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 50 * 50, 512)  # Adjust based on output size after conv layers
        self.fc2 = nn.Linear(512, 36)  # Output layer for 36 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 50 * 50)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load(r'cnn_model_20_epochs.pth'))
model.eval()

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image)

def predict_image(image):
    image = preprocess_image(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return categories[predicted.item()]

# Streamlit app
sl.title("American Sign Language Image Classification")

# Camera input
# camera_input = sl.camera_input("Take a picture")

# File upload input
uploaded_file = sl.file_uploader("Or choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])


if uploaded_file is not None:
    try:
        # Open and process the image
        image = Image.open(uploaded_file).convert('RGB')
        sl.image(image, caption='Uploaded Image', use_column_width=True)
        sl.write("Classifying...")

        # Predict
        prediction = predict_image(image)
        sl.write(f'Prediction: {prediction}')

    except Exception as e:
        sl.error(f"Error: {e}")

sl.write("Upload an image to classify.")
