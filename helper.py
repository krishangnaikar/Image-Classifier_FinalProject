#!/usr/bin/env python3
# PROGRAMMER: Krishang Naikar
# DATE CREATED: 10/23/2020                                  
# REVISED DATE: 

# Imports here

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import time
import helper
from workspace_utils import active_session
from collections import OrderedDict
import json
import argparse
from PIL import Image

arch_ins = {"vgg16":25088,
        "densenet121":1024
        }

def get_input_args():
    """
    This function parses the command line arguments
    Command Line Arguments:
      1. Train Image Folder as --data_dir 
      2. Folder to save the training data as --save_dir with default value current working dir
      2. Model Architecture as --arch with default value 'vgg'
      3. Hyperparameters as --learning_rate 0.001 --hidden_units 512 --epochs 20 with respective defaults
      4. Use of device as --gpu with default as "cpu"
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    

        
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers', required=True,
                    help = 'path to the folder of training images eg: flowers') 
    parser.add_argument('--save_dir', type = str, default ='./checkpoint.pth',
                    help = 'path to the folder and file_name to save the trained model')
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'Which CNN architecture you want to use? -- densenet121, alexnet, or vgg16') 
    parser.add_argument('--learning_rate', type = float, default = '0.001', 
                    help = 'learning rate for Gradient descent')
    parser.add_argument('--hid_units1', type = int, default = '1024', 
                    help = '# of input units for hidden unite#1') 
    parser.add_argument('--hid_units2', type = int, default = '512', 
                    help = '# of input units for hidden unite#2') 
    parser.add_argument('--epochs', type = int, default = '5', 
                    help = 'epochs times to run the training model')
    parser.add_argument('--device', type = str, default = 'cpu', 
                    help = 'device the model to run on i.e GPU/CPU')
    parser.add_argument('--dropout', type = float, default = '0.5', 
                    help = 'Drop out for the model') 


    in_arg = parser.parse_args()
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return in_arg


def data_loader(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                          ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                          ])

    # TODO: Load the datasets with ImageFolder
    #image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data.class_to_idx


def network(arch='vgg16', dropout=0.5, hidden_layer1 = 1024, hidden_layer2 = 512, lr = 0.001, device='cpu'):
    
    # load the model based on the selected Model Architecture
    if arch == 'vgg16':
        #print("Arch", arch)
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        #print("Arch", arch)
        model = models.densenet121(pretrained=True)
    else :
        print("ERROR: Please use one of these Architecture only -- densenet121, or vgg16")
    
    #print("Initial Model:", model)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch_ins[arch],hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('n_out1',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                          ('relu2', nn.ReLU()),
                          ('n_out2',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier

    #model.class_to_idx = train_data.class_to_idx
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
        #model.to(device);
        
    return model, criterion, optimizer
    
def train_model(model, criterion, optimizer, epochs, trainloader, validloader, device):
    
    train_losses, valid_losses = [], []
        
    with active_session():
        # do long-running work here
        for e in range(epochs):
            #print("Starting epoch #", e)
            running_loss = 0
        
            start = time.time()
            for images, labels in trainloader:
                if torch.cuda.is_available() and device =='gpu':
                    images, labels = images.to('cuda'), labels.to('cuda')
            
                optimizer.zero_grad()
            
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
            
            else:
                valid_loss = 0
                valid_accuracy = 0
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    model.eval()
                    for images, labels in validloader:
                    
                        if torch.cuda.is_available() and device =='gpu':
                            images, labels = images.to('cuda'), labels.to('cuda')
                            model.to('cuda')
                    
                        log_ps = model(images)
                        valid_loss += criterion(log_ps, labels)
                    
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor))
            
                model.train()
            
                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))
    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format((valid_accuracy/len(validloader))*100),
                       f"Time per-epoch: {(time.time() - start):.3f} seconds")
    
def save_checkpoint(model, path , arch, hidden_layer1, hidden_layer2, dropout, lr, epochs, class_to_idx):

    model.class_to_idx =  class_to_idx
    model.cpu
    torch.save({'arch' :arch,
                'hidden_layer1':hidden_layer1,
                'hidden_layer2':hidden_layer2,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)    
    print("Done Saving the CheckPoint as:", path)
    
    #Quick Validation
    state_dict = torch.load(path)
    #print(state_dict.values())
    print(state_dict.keys())
    
def load_checkpoint(filepath, device):
    
    if device == 'gpu':
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location="cpu")
        
        
    arch = checkpoint['arch']
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    dropout = checkpoint['dropout']
    learning_rate = checkpoint['lr']
    epochs = checkpoint['nb_of_epochs']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    model, criterion, optimizer = network(arch, dropout, hidden_layer1, hidden_layer2, learning_rate)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    
    # Using the shoter methods than manual steps as in part1
    
    proc_img = Image.open(image)

    process_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    processed_image = process_img(proc_img)
    return processed_image
    
    
def predict_image(image_file, model, topk, class_name, device):
    
    
    with open(class_name, 'r') as f:
        cat_to_name = json.load(f)
    
    
    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda')
        
    proces_image = process_image(image_file)
    #image = torch.from_numpy(proces_image).type(torch.FloatTensor)
    
     #used to make size of torch as expected. as forward method is working with batches,
    #doing that we will have batch size = 1 
    image = proces_image.unsqueeze (dim = 0) 
    
    
    with torch.no_grad():
        model.eval()
        
        if device == 'gpu':
            log_ps = model(image.cuda())
        else:
            log_ps = model(image)
            
        ps = torch.exp(log_ps)
        
        top_probabilities, top_classes = ps.topk(topk)
        #print("top_probabilities", top_probabilities)
        #print("top_classes", top_classes)
        
        top_probabilities_list = top_probabilities.tolist()[0]
        top_classes_list = top_classes.tolist()[0]
        
        #print("top_probabilities_list", top_probabilities_list)
        #print("top_classes_list", top_classes_list)
        
        
        idx_to_class  = {val: key for key, val in model.class_to_idx.items() }
        #print("cat_to_name[i]:", cat_to_name['91'])
        #print("idx_to_class ", idx_to_class )
        #print(type(top_classes_list))
        #print("model.class_to_idx.items()", model.class_to_idx)
        
        top_label = [idx_to_class[i] for i in top_classes_list]
        top_flower = [cat_to_name[str(i)] for i in top_label]
        
        return top_probabilities_list, top_classes_list, top_label, top_flower