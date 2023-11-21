import torch
import torch.nn as nn

def loss_function(recon_x, x, mu, std):
    
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
