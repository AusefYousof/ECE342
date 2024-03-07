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



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((144, 174)),  # Resize to target dimensions
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize grayscale images; adjust as needed
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Ensure this creates a 3-channel image
])





dataset_path = r"C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\human detection dataset"  


train_dataset = torchvision.datasets.ImageFolder(dataset_path + r'\train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(dataset_path + r'\val', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(dataset_path + r'\test', transform=transform)


total = len(train_dataset)+len(val_dataset)+len(test_dataset)
perc_train = str(round(len(train_dataset)/total * 100,2))
perc_val = str(round(len(val_dataset)/total * 100,2))
perc_test = str(round(len(test_dataset)/total * 100,2))
print("Testing Data is:", perc_train + "% train,", perc_val + "% validation,", perc_test + "% test\n") 
#looking for about a 75 - 12.5 - 12.5 percent split


alexnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    alexnet = alexnet.cuda()

#probably gonna go unused causing pathing is so bad
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


#define our classes (traffic signals)
classes = ['0', '1']

path = r"C:\Users\Ausef Yousof\OneDrive\Documents\ECE YEAR 3 SEM 2\ECE342\project\AlexNet\Features"

#function to extract alexnet features from our data, save to folder.
def alex_features(data):
    if data == "train":
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    elif data == "val":
        data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    elif data == "test":
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    else:
        print("Error, invalid dataset name")
        return None
    #iterate through loaded data, save alex net features
    n = 0
    for imgs, labels in iter(data_loader):
        features = alexnet.features(imgs)
        features_tensor = torch.from_numpy(features.detach().numpy())
        torch.save(features_tensor.squeeze(0), path + '\\'+ data + '\\' + classes[labels] + '\/' + 'feature_bs1_' + str(n) + '.tensor')
        n += 1

#get features
alex_features("train")
alex_features("val")
alex_features("test")

#load our features

train_features = torchvision.datasets.DatasetFolder(path + r"\train", loader=torch.load,
                                                    extensions=('.tensor'))
val_features = torchvision.datasets.DatasetFolder(path + r"\val", loader=torch.load,
                                                  extensions=('.tensor'))
test_features = torchvision.datasets.DatasetFolder(path + r"\test", loader=torch.load,
                                                   extensions=('.tensor'))

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=2): # 0 or 1 for person or no person
        super(CustomAlexNet, self).__init__()
        self.name = "CustomAlexNet"
        # Feature extraction part (could use alexnet.features if dimensions match)
        self.features = nn.Sequential(
            # Define your conv layers here if needed, or use pre-trained ones
        )

        # The size of the flattened feature tensor
        self.flattened_size = 256 * 3 * 4  
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Example size
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.flattened_size)  # Flatten the features
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def get_accuracy_ALEX(model, batch_size, train=False):
    if train:
        data = train_features
    else:
        data = val_features
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=batch_size):


        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################


        output = model(imgs)

        # Select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total



trainpath = "C:\\Users\\Ausef Yousof\\OneDrive\\Documents\\ECE YEAR 3 SEM 2\\ECE342\\project\\training"
def train_ALEX(model, data, batch_size=64, learning_rate=0.05, num_epochs=25):
    torch.manual_seed(1000)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    # Training
    start_time = time.time()
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):


            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################


            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # Save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            n += 1
        train_acc.append(get_accuracy_ALEX(model, batch_size=batch_size, train=True)) # compute training accuracy
        val_acc.append(get_accuracy_ALEX(model, batch_size=batch_size, train=False))  # compute validation accuracy

        # Save the current model (checkpoint) to a file
        #model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        #model_path = os.path.join(trainpath, model_path)
        #torch.save(model.state_dict(), model_path)

    

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    #save the model
    file_path = os.path.join(trainpath,model.name)
    torch.save(model, file_path)


    # Plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(range(1 ,num_epochs+1), train_acc, label="Train")
    plt.plot(range(1 ,num_epochs+1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))



model = CustomAlexNet()

#first training with parameters listed above
#train_ALEX(model, train_features, batch_size=32, learning_rate=0.01, num_epochs=15)
#dont run if already trained and saved a model



load_model = torch.load('C:/Users/Ausef Yousof/OneDrive/Documents/ECE YEAR 3 SEM 2/ECE342/project/training/CustomAlexNet')
model.eval()
print("done")







#demoing model


demo_dataset = torchvision.datasets.ImageFolder('C:/Users/Ausef Yousof/OneDrive/Documents/ECE YEAR 3 SEM 2/ECE342/project/training/demoimages', transform=transform)

#extract test images features
def alex_features_demo(data="demo"):
    data_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=1, shuffle=True)


    #iterate through loaded data, save alex net features
    n = 0
    for imgs, labels in iter(data_loader):
        features = alexnet.features(imgs)
        features_tensor = torch.from_numpy(features.detach().numpy())
        torch.save(features_tensor.squeeze(0), path + '\\'+ "demo" + '\\' + classes[labels] + '\\' + 'feature_bs1_' + str(n) + '.tensor')
        n += 1

#get demo image features for input to model
alex_features_demo()

demo_features = torchvision.datasets.DatasetFolder(path + "\\demo", loader=torch.load,
                                                    extensions=('.tensor'))

demo_loader = torch.utils.data.DataLoader(demo_features, batch_size=1, shuffle=True)

for imgs, labels in iter(demo_loader):
        
    with torch.no_grad():
        print("Ground truth:", labels)
        print("\n")
        output = load_model(imgs)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        print("Predicted class:", predicted_class.item())

#print(output)