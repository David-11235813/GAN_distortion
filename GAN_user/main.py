import os
import time
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import *

latent_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

G = Generator().to(device)


mid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'middleman_folder'))
models_dir = os.path.join(mid_dir, 'models')
latest_model = get_last_subfolder(models_dir)

G.load_state_dict(torch.load(os.path.join(latest_model, "generator.pth"), map_location=device))
G.eval()

# Generate one image
z = torch.randn(1, latent_dim, device=device)
fake_img = G(z).view(1, 28, 28)


# Save to disk
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))
os.makedirs(models_dir, exist_ok=True)

img_name = f"generated_{time.strftime("%Y%m%d-%H%M%S")}.png"
save_image(fake_img, os.path.join(models_dir, img_name), normalize=True)
print(f"Image saved as {img_name}")
