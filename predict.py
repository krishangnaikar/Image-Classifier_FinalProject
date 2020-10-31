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

import argparse

arch_ins = {"vgg16":25088,
        "densenet121":1024
        }

# Main program function defined below
def main():
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    

        
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--image_file', type = str, default = 'flowers/test/9/image_06413.jpg', required=True,
                    help = 'Input Img file name to predict eg: flowers/test/9/image_06413.jpg') 
    parser.add_argument('--load_chkpt', type = str, default ='./checkpoint.pth',
                    help = 'path to the folder and file_name to load the pre-trained model')
    parser.add_argument('--device', type = str, default = 'cpu', 
                    help = 'device the model to run on i.e GPU/CPU')
    parser.add_argument('--topk', type = int, default = '5', 
                    help = 'topk probabilites to display')
    parser.add_argument('--class_name', type = str, default = 'cat_to_name.json', 
                    help = 'JSON file that maps the class values to other category names')

    predict_arg = parser.parse_args()
    
    print("predict_arg:\n", predict_arg)
    
    model = load_checkpoint(predict_arg.load_chkpt, predict_arg.device)
    print("Loaded Model:", model)
    
    #process_image(predict_arg.image_file)
    
    top_prob, top_classes, top_label, top_flower = predict_image(predict_arg.image_file, model, predict_arg.topk, predict_arg.class_name, predict_arg.device)
    
    print("Top Probabilities:\n", top_prob)
    print("Top Classes:\n", top_classes)
    print("Top Labels:\n", top_label)
    print("Top Flowers:\n", top_flower)
# Call to main function to run the program
if __name__ == "__main__":
    main()    