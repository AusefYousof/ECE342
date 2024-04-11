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
import http.server
import socketserver
import webbrowser
import threading
import subprocess
import platform


import tkinter as tk
from tkinter import messagebox

from torchvision.transforms.functional import to_pil_image


#this code is going to be moved into serial_monitor_lab_06.py and .exe will be generated because
#this method of sending the photo to the script which loads and predicts everytime is not efficient

#code was generated as a exe and in serial_monitor_lab_06_final


def show_detection_alert():
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showinfo("Detection Alert", "PERSON DETECTED!")  # show an "Info" message box
    root.destroy()

####################
#WINDOW STUFF
####################



def openhtmlfile(filepath):
    webbrowser.open('file://' + filepath)
    time.sleep(2)  # Wait for 2 seconds
    #subprocess.Popen('taskkill /f /im chrome.exe', shell=True)
    
    


class PD_CNN(nn.Module):
    def __init__(self):
        super(PD_CNN, self).__init__()
        # Increase convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive Pooling to handle varying dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        # Revise fully connected layers according to new output size
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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


    #image_pil = to_pil_image(input_tensor)

    # Display the image and print its label
    #plt.imshow(image_pil, cmap='gray')
    #plt.show()
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
    html_file_path = r'alert.html'
    if prediction:
        openhtmlfile(html_file_path)

    
    
    # Print the prediction result
    print(prediction)
    