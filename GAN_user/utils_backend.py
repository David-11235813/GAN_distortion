import os
import torch
import torch.nn as nn
from torchvision.utils import save_image


latent_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_last_subfolder(parent_dir):
    subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs: return None
    return os.path.join(parent_dir, sorted(subdirs)[-1])


def LoadModel():
    mid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'middleman_folder'))
    models_dir = os.path.join(mid_dir, 'models')
    latest_model = get_last_subfolder(models_dir)

    gen = Generator().to(device)
    gen.load_state_dict(torch.load(os.path.join(latest_model, "generator.pth"), map_location=device))
    gen.eval()
    return gen


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


def GenerateImage(G, output_dir):
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        fake_img = G(z).view(1, 1, 28, 28)

    last_img_path = os.path.join(os.path.dirname(output_dir), 'temp.png')
    save_image(fake_img, last_img_path, normalize=True)

    return [fake_img, last_img_path]
