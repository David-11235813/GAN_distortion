import torch
import torch.nn as nn
import torch.optim as optim

from __autoencoder import Encoder, Decoder
from img_display_utils import display_autoencoder_reconstructions
from my_filesystem import prepare_model_save_path


device='cuda' if torch.cuda.is_available() else 'cpu'

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


def save_autoencoder(autoencoder:Autoencoder, directory:list[str], filename:str, new_timedate:bool) -> str:
    save_path = prepare_model_save_path(directory, filename, new_timedate)
    torch.save({
        'state_dict': autoencoder.state_dict(),
        'latent_dim': autoencoder.latent_dim
    }, save_path)
    return save_path


def load_autoencoder(autoencoder_filepath:str) -> Autoencoder:
    checkpoint = torch.load(autoencoder_filepath, map_location=device)
    #print(checkpoint['latent_dim'])    #prints the saved _compression_size value, e.g. 16 or 32

    autoencoder = Autoencoder(latent_dim=checkpoint['latent_dim']).to(device)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    autoencoder.eval()
    return autoencoder


def show_autoencoder_reconstruction(autoencoder, dataloader):
    autoencoder.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        reconstructions, features = autoencoder(images)

        display_autoencoder_reconstructions(images, reconstructions, 10)
        print(f"Feature vector size: {features.shape}")
