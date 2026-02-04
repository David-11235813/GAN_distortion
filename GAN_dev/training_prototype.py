import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import time


# ---- Hyperparameters ----
latent_dim = 100
batch_size = 64
epochs = 1
lr = 0.0002

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Dataset (MNIST) ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')
)

dataset = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- Models ----
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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

# ---- Training setup ----
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr)
opt_D = torch.optim.Adam(D.parameters(), lr=lr)

# ---- Training loop ----
for epoch in range(epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.view(-1, 784).to(device)

        # Labels
        real_labels = torch.ones(real_imgs.size(0), 1, device=device)
        fake_labels = torch.zeros(real_imgs.size(0), 1, device=device)

        # ---- Train Discriminator ----
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        fake_imgs = G(z)

        loss_D = (
            criterion(D(real_imgs), real_labels) +
            criterion(D(fake_imgs.detach()), fake_labels)
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train Generator ----
        loss_G = criterion(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D loss: {loss_D.item():.3f} | G loss: {loss_G.item():.3f}")

# ---- Save generator ----



models_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'middleman_folder', 'models')
)
os.makedirs(models_dir, exist_ok=True)

run_dir = os.path.join(models_dir, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

torch.save(G.state_dict(), os.path.join(run_dir, 'generator.pth'))
torch.save(D.state_dict(), os.path.join(run_dir, 'discriminator.pth'))

print(f"Generator saved to {run_dir}")
