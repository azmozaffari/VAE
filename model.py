import torch
import torchvision
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self,size_x,size_y):
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear()

        # decoder
    
    def encoder(self,x):
        
        