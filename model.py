
import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self,size_x,size_y, h1_size,h2_size):
        super(VAE, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.size_x = size_x
        self.size_y = size_y

        # encoder
        self.fc1_e = nn.Linear(size_x*size_y, h1_size)
        self.fc2_e = nn.Linear(h1_size,h2_size)
        self.fc3_mean = nn.Linear(h2_size,h2_size)
        self.fc3_std = nn.Linear(h2_size,h2_size)
        

        # decoder
        self.fc1_d = nn.Linear(h2_size, h1_size)
        self.fc2_d = nn.Linear(h1_size,size_x*size_y)

        # non-linear activation function
        self.relu = nn.ReLU()



    
    def encoder(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size,-1)
        x = self.relu(self.fc1_e(x))
        x = self.relu(self.fc2_e(x))
        mu = self.fc3_mean(x)
        std = self.fc3_std(x)
        return mu, std
    
    
    def randomSampling(mu, std):

        std = torch.exp(0.5*std)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        

    def decoder(self, z):
        
        # z = z.view(-1, self.h2_size)
        z = self.relu(self.fc1_d(z))
        z = self.relu(self.fc2_d(z))
        batch_size = z.size(0)
        z = z.view(batch_size,1,self.size_x, self.size_y)
        return z
    

    


