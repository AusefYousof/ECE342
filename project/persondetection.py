import torch, torchvision, os, torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.models #gonna use AlexNet
from torchvision.models.alexnet import AlexNet_Weights
import time

use_cuda = True; # try to use CUDA when training model
dimensions = [144,174]

###########################################################################################################
##############################            DATA PROCESSING              ####################################
###########################################################################################################

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((144, 174)),  # Resize to target dimensions
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Adjusted for a single channel
    ])


train_dataset_path =  r'C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\human detection dataset\train'
val_dataset_path =  r'C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\human detection dataset\val'
test_dataset_path =  r'C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\human detection dataset\test'

train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(val_dataset_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(test_dataset_path, transform=transform)



total = len(train_dataset)+len(val_dataset)+len(test_dataset)
perc_train = str(round(len(train_dataset)/total * 100,2))
perc_val = str(round(len(val_dataset)/total * 100,2))
perc_test = str(round(len(test_dataset)/total * 100,2))
#print("Testing Data is:", perc_train + "% train,", perc_val + "% validation,", perc_test + "% test\n") 
#looking for about a 75 - 12.5 - 12.5 percent split


#probably gonna go unused causing pathing is so bad
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path



###########################################################################################################
###############################            NETWORK DEFINITION          ####################################
###########################################################################################################



#define our classes (traffic signals)
classes = ['0', '1']

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





###########################################################################################################
##################################             TRAINING           #########################################
###########################################################################################################
    
trainpath = "training"

def normalize_label(labels):
    """
    Given a tensor containing 2 possible values, normalize this to 0/1

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)/(max_val - min_val)
    return norm_labels


def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        labels = labels.unsqueeze(1)
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.5).squeeze().long() != labels.squeeze().long()  # Adjusted comparison for binary classification
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss


def train(net, batch_size=8, learning_rate=0.001, num_epochs=15):
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            labels = normalize_label(labels) # Convert labels to 0/1
            labels = labels.unsqueeze(1)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        #model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        #torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    
    #save model
    torch.save(net, "project\serial_monitor\serial_monitor_lab_06/PD_CNN")



model = PD_CNN()
if use_cuda and torch.cuda.is_available():
    model = model.cuda()
else:
    print("Not using Cuda \n")

train(model)

#model is trained at this point, load it in modelpreds.py




