
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
        log_var = self.fc3_std(x)

        
        return mu, log_var
    
    
    def randomSampling(self, mu, log_var):

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(log_var)
        samples = eps.mul(std).add_(mu)
        
        return  samples# return z sample
        

    def decoder(self, z):
       
        
        z = self.relu(self.fc1_d(z))
        z = torch.sigmoid(self.fc2_d(z))
        
        return z
    
    
    def forward(self, img):
        mu, log_var = self.encoder(img)
        
        z = self.randomSampling(mu, log_var)
        r_img = self.decoder(z)
        return r_img, mu, log_var
    

   