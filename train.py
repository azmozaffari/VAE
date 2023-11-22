import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import random
import os

from model import * 
from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(" the code is running on: ", device)
SEED = 42

def deterministic(seed):
    """
    Setup execution state so that we can reproduce multiple executions.
    Make the execution "as deterministic" as possible.

    random_seed: seed used to feed torch, numpy and python random
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)      
       
    
deterministic(SEED)



lr = 1e-3 
batch_size = 16
model = VAE(28,28,400,20)

model = model.to(device)
epochs = 101


optimizer = optim.Adam(model.parameters(), lr)

img_transform = transforms.Compose([
    transforms.ToTensor()
    ])

dataset = MNIST('./data', transform=img_transform, train = True, download=True)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

def train(epochs,  model,dataloader,optimizer):
    
    
    # make the directory to save the models
    if (not os.path.exists("./models")):
        os.mkdir("./models")

    for epoch in range(epochs):
        total_loss = 0
        for batch_id, data in enumerate(dataloader):
            img, label = data
            img = img.to(device)


          
            
            r_img, mu,log_var = model(img)
            img = img.squeeze()
            row = img.size(1)
            col = img.size(2)
            

            
            loss = loss_function(r_img, img.view(-1,row*col), mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 

            
        # print the average loss per epoch
        n_samples = batch_id* batch_size
        print("epoch:", epoch,  "loss is", total_loss/n_samples)
    

        # save the model in every 50 epochs
            
        if epoch%50 == 0:            
            
            torch.save(model.state_dict(), './models/vae_model'+str(epoch)+'.pth')

            

            
        

if __name__ == "__main__":

    train(epochs,  model, dataloader, optimizer)


