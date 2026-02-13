import torch
import torch.nn as nn
import torch.optim as optim

from __utilise_dataset import get_default_dataloader, default_dataset_path, test_dataloader
from img_display_utils import display_autoencoder_reconstructions, reconstruct_image_and_display_features

device='cuda' if torch.cuda.is_available() else 'cpu'

# Encoder: Compresses image to feature vector
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # latent_dim is how many numbers we compress the image into

        self.encoder = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3,   64,  kernel_size=4, stride=2, padding=1),  # ->  64 x 64 x 64
            nn.ReLU(),

            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> 512 x  8 x  8
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
    def __init__(self, latent_dim):
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

            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,  3,   kernel_size=4, stride=2, padding=1),  # -> 3 x 128 x 128
            nn.Tanh()  # Output in range [-1, 1] to match our normalized images
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)  # Reshape to spatial dimensions
        return self.decoder(x)


# Complete Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


# Training function
def train_autoencoder(dataloader, compression_size=32, epochs=30):
    autoencoder = Autoencoder(latent_dim=compression_size).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)
    criterion = nn.L1Loss()  # Measures how different reconstruction is from original

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)

            # Forward pass
            reconstructions, _ = autoencoder(images) # calls: autoencoder.forward(images) - but safer.
            loss = criterion(reconstructions, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return autoencoder



def load_autoencoder(autoencoder_filepath):
    gen = Autoencoder().to(device)
    gen.load_state_dict(torch.load(autoencoder_filepath, map_location=device))
    gen.eval()
    return gen




def load_and_test_model(model_path:str, howmany=1):
    dataloader = get_default_dataloader(default_dataset_path)
    model = load_autoencoder(model_path)
    for _ in range(howmany):
        show_autoencoder_reconstruction(model, dataloader)


# Visualization function
def show_autoencoder_reconstruction(model, dataloader, device=device):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        reconstructions, features = model(images)

        display_autoencoder_reconstructions(images, reconstructions, 10)

        print(f"Feature vector size: {features.shape}")



def display_reconstructed_features(model, dataloader, device=device):
    model.eval()
    with torch.no_grad():
        image1, _ = next(iter(dataloader))[0].unsqueeze(0)  # First image; shape: [1, 3, 128, 128] (model expects [batch_size, C, H, W])
        image1 = image1.to(device)
        reconstruct_image_and_display_features(model, image1)
