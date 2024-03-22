import torch
import sys
import pickle
from PIL import Image
import io
import torchvision.transforms as transforms
import torch, torchvision, os, torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.models.alexnet import AlexNet_Weights
import time



class PD_CNN(nn.Module):
    def __init__(self):
        super(PD_CNN, self).__init__()
        self.name = "PD_CNN"
        # Use fewer filters in the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)  # From 32 to 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # From 64 to 32
        # Adjust the size of the fully connected layer
        self.fc1 = nn.Linear(32 * 36 * 43, 256)  # Reduced size
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 36 * 43)  # Adjust the flattening
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_model(model_path="PD_CNN"):
    model = torch.load(model_path)
    model.eval()  
    return model

def predict(model, img_bytes):
    """
    Make a prediction for an input image.
    """
    # Convert bytes back to PIL Image
    size = (144,174)
    img = Image.frombytes('L', size, img_bytes)
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((144, 174)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    # Apply transformations
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # Convert output probabilities to predicted class (0 or 1)
    pred = torch.sigmoid(output).item() > 0.5
    return pred

if __name__ == "__main__":
    # Load the trained model
    model = load_model()
    
    # Read image bytes from stdin
    img_bytes = sys.stdin.buffer.read()
    
    # Predict
    prediction = predict(model, img_bytes)
    
    # Print the prediction result
    print(prediction)

