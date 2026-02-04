import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'

# Encoder: Compresses image to feature vector
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # latent_dim is how many numbers we compress the image into

        self.encoder = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Flatten to vector
            nn.Flatten(),  # -> 512 * 8 * 8 = 32768
            nn.Linear(512 * 8 * 8, latent_dim)  # -> latent_dim
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder: Reconstructs image from feature vector
class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        # Expand from vector back to spatial dimensions
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            # Input: 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # -> 3 x 128 x 128
            nn.Tanh()  # Output in range [-1, 1] to match our normalized images
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)  # Reshape to spatial dimensions
        return self.decoder(x)


# Complete Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


# Training function
def train_autoencoder(model, dataloader, epochs=10, device=device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()  # Measures how different reconstruction is from original

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)

            # Forward pass
            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model



def LoadAutoencoder():
    gen = Autoencoder().to(device)
    gen.load_state_dict(torch.load(os.path.join("autoencoder.pth"), map_location=device))
    gen.eval()
    return gen

# Visualization function
def show_reconstruction(model, dataloader, device=device):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        reconstructions, features = model(images)

        # Show original vs reconstruction
        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            # Original
            orig = images[i].cpu().permute(1, 2, 0).numpy()
            orig = (orig + 1) / 2  # Denormalize
            axes[0, i].imshow(orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstruction
            recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
            recon = (recon + 1) / 2  # Denormalize
            axes[1, i].imshow(recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.show()

        print(f"Feature vector size: {features.shape}")