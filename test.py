from model import VAE
import torch
from torchvision.utils import save_image

PATH = './models/vae_model100.pth'

model = VAE(28,28,400,20)
model.load_state_dict(torch.load(PATH))
model.eval()

eps = torch.normal(0, 1, size=(10, 20))


rec_img = model.decoder(eps)
rec_img = rec_img.view(-1,28,28)
rec_img = torch.unsqueeze(rec_img, 1)

print(rec_img.size())
save_image(rec_img, './recon_imgs/image.png')