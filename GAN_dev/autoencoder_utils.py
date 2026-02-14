import torch
import torch.nn as nn
import torch.optim as optim

from img_display_utils import display_autoencoder_reconstructions
from my_filesystem import prepare_model_save_path


device='cuda' if torch.cuda.is_available() else 'cpu'


# Encoder: Compresses image to feature vector
class Encoder(nn.Module):
    def __init__(self, latent_dim, channels_nr):
        super().__init__()
        # latent_dim is how many numbers we compress the image into

        self.encoder = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(channels_nr,   64,  kernel_size=4, stride=2, padding=1),  # ->  64 x 64 x 64
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
    def __init__(self, latent_dim, channels_nr):
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

            nn.ConvTranspose2d(64,  channels_nr,   kernel_size=4, stride=2, padding=1),  # -> 3 x 128 x 128
            nn.Tanh()  # Output in range [-1, 1] to match our normalized images
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)  # Reshape to spatial dimensions
        return self.decoder(x)


# Complete Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, is_grayscale):
        super().__init__()
        self.latent_dim = latent_dim
        self.is_grayscale = is_grayscale
        channels_nr = 3 if not self.is_grayscale else 1

        self.encoder = Encoder(latent_dim, channels_nr)
        self.decoder = Decoder(latent_dim, channels_nr)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


# Training function
def train_autoencoder(dataloader, compression_size:int, epochs=30) -> Autoencoder:
    autoencoder = Autoencoder(latent_dim=compression_size, is_grayscale=dataloader.is_grayscale).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)
    criterion = nn.L1Loss()  # Measures how different reconstruction is from original

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)

            # Forward pass
            reconstructions, _ = autoencoder(images) # calls: autoencoder.forward(images), but safer.
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
        'latent_dim': autoencoder.latent_dim,
        'is_grayscale': autoencoder.is_grayscale
    }, save_path)
    return save_path


def load_autoencoder(autoencoder_filepath:str) -> Autoencoder:
    checkpoint = torch.load(autoencoder_filepath, map_location=device)

    #print(checkpoint['latent_dim'])    #prints the saved _compression_size value, e.g. 16 or 32
    latent_dim = checkpoint['latent_dim']
    is_grayscale = checkpoint['is_grayscale'] if 'is_grayscale' in checkpoint else False #backwards-compatibility

    autoencoder = Autoencoder(latent_dim=latent_dim, is_grayscale=is_grayscale).to(device)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    autoencoder.eval()
    return autoencoder


def show_autoencoder_reconstruction(autoencoder, dataloader):
    autoencoder.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        reconstructions, features = autoencoder(images)

        display_autoencoder_reconstructions(images, reconstructions, 6)
        print(f"Feature vector size: {features.shape}")


def show_encoder_output(autoencoder, dataloader):
    encoder = autoencoder.encoder
    encoder.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        features = encoder(images)
        print(f"Feature vector size: {features.shape}")
        return features