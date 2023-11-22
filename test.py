from model import VAE
import torch
from torchvision.utils import save_image

PATH = './models/vae_model100.pth'

model = VAE(28,28,400,20)
model.load_state_dict(torch.load(PATH))
model.eval()

# genereate normal gaussian noise
eps = torch.normal(0, 1, size=(10, 20))

grid = torch.zeros(1,20)
grid[0,19] = 0.05
grid[0,10] = 0.05
grid[0,1] = 0.05

ln = torch.zeros(64,20) + 0.2

for i in range(64):
    ln[i,:] = ln[i,:] + i*grid





# call the decoder to generate samples from the normal disribution
rec_img = model.decoder(ln)
rec_img = rec_img.view(-1,28,28)
rec_img = torch.unsqueeze(rec_img, 1)

print(rec_img.size())
save_image(rec_img, './recon_imgs/image.png')