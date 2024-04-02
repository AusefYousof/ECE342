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
import time

import tkinter as tk
from tkinter import messagebox


#this code is going to be moved into serial_monitor_lab_06.py and .exe will be generated because
#this method of sending the photo to the script which loads and predicts everytime is not efficient


def show_detection_alert():
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showinfo("Detection Alert", "PERSON DETECTED!")  # show an "Info" message box
    root.destroy()




class PD_CNN(nn.Module):
    def __init__(self):
        super(PD_CNN, self).__init__()
        self.name = "PD_CNN"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1_input_size = 8 * (174//2) * (144//2)  
        self.fc1 = nn.Linear(self.fc1_input_size, 128) 
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.fc1_input_size)  
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
    size = (174,144)
    
    img = Image.frombytes('L', size, img_bytes)

    #############################################
    #to save images (making dataset from scratch)
    ##############################################
    
    #save_dir = "1"
    #to save images (making dataset from scratch)
    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    #image_path = os.path.join(save_dir, f"image_{timestamp}.jpg")
    #img.save(image_path)

    #to output the image we received from stdin
    #img.show()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
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
    #DETECTED! window if person detected
    if prediction:
        show_detection_alert()

    
    
    # Print the prediction result
    print(prediction)

    

