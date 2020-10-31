

# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import os
import sys
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
from helper import *



# Main program function defined below
def main():

    in_arg = get_input_args()
    print("in_args:\n", in_arg)
    
    trainloader, validloader, testloader, class_to_idx = data_loader(in_arg.data_dir)
    #print("class_to_idx:", class_to_idx)
    #print(type(trainloader))
    #print(len(trainloader))
    
    model, criterion, optimizer = network(in_arg.arch, in_arg.dropout, in_arg.hid_units1, in_arg.hid_units2, in_arg.learning_rate, in_arg.device)
    
    print("Loaded Model:", model.classifier)
    
    #Training the Model
    train_model(model, criterion, optimizer, in_arg.epochs, trainloader, validloader, in_arg.device)
    
    #Saveing the Model
    save_checkpoint(model, in_arg.save_dir, in_arg.arch, in_arg.hid_units1, in_arg.hid_units2, in_arg.dropout, in_arg.learning_rate, in_arg.epochs, class_to_idx)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
