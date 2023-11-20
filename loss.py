import torch
import torch.nn as nn

def loss_function(recon_x, x, mu, std):
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    
    # KL divergence
    
    return BCE + KLD


def 