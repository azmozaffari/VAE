import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os



batch_size = 16


img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)



