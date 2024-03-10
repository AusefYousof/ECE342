import torch
import torch.nn as nn
import torch.nn.functional as F

class PD_CNN(nn.Module):
    def __init__(self):
        super(PD_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 36 * 43, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36 * 43)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    

# Make sure the model class is defined as above
model = PD_CNN()

# If using GPU for inference
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load the trained model
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.to(device)

model.eval()  # Set the model to evaluation mode

from torchvision import transforms
from PIL import Image

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((144, 174)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load and transform an image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img_transformed = transform(img)
    return img_transformed.unsqueeze(0) 


def predict(image_path, model):
    img = preprocess_image(image_path)
    img = img.to(device)  # Move to device
    with torch.no_grad():
        output = model(img)
        prediction = output.round().item()  
    return prediction


image_path = 'C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\training\demoimages\0\30.png'
prediction = predict(image_path, model)
print(f'Prediction: {prediction}')
