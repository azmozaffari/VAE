import torch
import torch.nn as nn

def loss_function(r_x, x, mu, log_var):
    
    reconstruction_function = nn.MSELoss(size_average=False)
 
    MSE = reconstruction_function(r_x, x)  # mse loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE + KLD
