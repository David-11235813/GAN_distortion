#temporary checks and proofs of concept here.

import my_filesystem
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from utilise_dataset import get_default_dataloader

from main import load_and_test_model_from_path

device='cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0),  # -> 1 x 1 x 1
            #nn.Flatten(),
            #nn.Sigmoid()  # Output probability: real (1) or fake (0)
        )

    def forward(self, x):
        return self.discriminator(x).view(-1, 1)


def train_gan_autoencoder(autoencoder, discriminator, dataloader, epochs=10, device=device):
    """
    This training loop implements the adversarial game between:
    - The autoencoder (generator) trying to create realistic reconstructions
    - The discriminator trying to spot which images are reconstructions
    """

    autoencoder = autoencoder.to(device)
    discriminator = discriminator.to(device)

    # Separate optimizers for generator and discriminator (standard GAN practice)
    optimizer_G = optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()  # For discriminator's real/fake classification
    reconstruction_loss = nn.MSELoss()  # We'll keep some reconstruction loss for stability

    # Weight for balancing adversarial vs reconstruction loss
    lambda_recon = 10  # You can tune this - higher means more emphasis on exact reconstruction

    for epoch in range(epochs):
        total_loss_G = 0
        total_loss_D = 0

        for images, _ in dataloader:
            batch_size = images.size(0)
            images = images.to(device)

            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ===========================
            # Train Discriminator
            # Goal: Maximize log(D(x)) + log(1 - D(G(x)))
            # In other words, correctly identify real as real and fake as fake
            # ===========================

            optimizer_D.zero_grad()

            # Loss on real images (discriminator should output ~1)
            real_output = discriminator(images)
            loss_D_real = adversarial_loss(real_output, real_labels)

            # Generate fake images (reconstructions)
            reconstructions, _ = autoencoder(images)

            # Loss on fake images (discriminator should output ~0)
            # .detach() is crucial - we don't want to update the generator yet
            fake_output = discriminator(reconstructions.detach())
            loss_D_fake = adversarial_loss(fake_output, fake_labels)

            # Combined discriminator loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # ===========================
            # Train Generator (Autoencoder)
            # Goal: Minimize log(1 - D(G(x))) or equivalently maximize log(D(G(x)))
            # Also maintain reconstruction quality
            # ===========================

            optimizer_G.zero_grad()

            # We already have reconstructions from above, but need new forward pass
            # because we detached earlier
            reconstructions, _ = autoencoder(images)

            # Adversarial loss: fool the discriminator (want discriminator to output ~1)
            fake_output = discriminator(reconstructions)
            loss_G_adv = adversarial_loss(fake_output, real_labels)  # Note: using real_labels!

            # Reconstruction loss: maintain similarity to original
            loss_G_recon = reconstruction_loss(reconstructions, images)

            # Combined generator loss
            loss_G = loss_G_adv + lambda_recon * loss_G_recon
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D = total_loss_D / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, G_Loss: {avg_loss_G:.4f}, D_Loss: {avg_loss_D:.4f}")

    return autoencoder, discriminator

def train_gan_autoencoder_v2(autoencoder, discriminator, dataloader, epochs=10, device=device):
    """
    This version includes several stability improvements for GAN training
    """

    autoencoder = autoencoder.to(device)
    discriminator = discriminator.to(device)

    # Optimizers with same settings as before
    optimizer_G = optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.MSELoss()

    # Start with high reconstruction emphasis, gradually reduce it
    # This gives the autoencoder time to learn basic reconstruction first
    lambda_recon_start = 30.0  # Start very high
    lambda_recon_end = 0.1  # End at a balanced value

    # Number of epochs to pretrain with only reconstruction loss
    pretrain_epochs = 10

    for epoch in range(epochs):
        # Gradually decrease reconstruction weight over training
        # This creates a curriculum: first learn to reconstruct, then learn to be realistic
        if epoch < pretrain_epochs:
            lambda_recon = lambda_recon_start
            train_discriminator = False  # Don't train discriminator during warmup
        else:
            progress = (epoch - pretrain_epochs) / max(1, epochs - pretrain_epochs)
            lambda_recon = lambda_recon_start + (lambda_recon_end - lambda_recon_start) * progress
            train_discriminator = True

        total_loss_G = 0
        total_loss_D = 0
        total_recon_loss = 0
        total_adv_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ===========================
            # Train Discriminator (only after warmup)
            # ===========================

            if train_discriminator:
                optimizer_D.zero_grad()

                real_output = discriminator(images)
                loss_D_real = adversarial_loss(real_output, real_labels)

                reconstructions, _ = autoencoder(images)
                fake_output = discriminator(reconstructions.detach())
                loss_D_fake = adversarial_loss(fake_output, fake_labels)

                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()

                total_loss_D += loss_D.item()

            # ===========================
            # Train Generator (Autoencoder)
            # ===========================

            optimizer_G.zero_grad()

            reconstructions, _ = autoencoder(images)

            # Always compute reconstruction loss
            loss_G_recon = reconstruction_loss(reconstructions, images)

            # Only add adversarial loss after warmup
            if train_discriminator:
                fake_output = discriminator(reconstructions)
                loss_G_adv = adversarial_loss(fake_output, real_labels)
                loss_G = loss_G_adv + lambda_recon * loss_G_recon
                total_adv_loss += loss_G_adv.item()
            else:
                # During warmup, only use reconstruction loss
                loss_G = loss_G_recon

            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            total_recon_loss += loss_G_recon.item()

        # Print detailed statistics to understand what's happening
        avg_loss_G = total_loss_G / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)

        if train_discriminator:
            avg_loss_D = total_loss_D / len(dataloader)
            avg_adv = total_adv_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | G_Loss: {avg_loss_G:.4f} | D_Loss: {avg_loss_D:.4f} | "
                  f"Recon: {avg_recon:.4f} | Adv: {avg_adv:.4f} | λ: {lambda_recon:.1f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} [WARMUP] | Recon: {avg_recon:.4f}")

    return autoencoder, discriminator



def use_gan_autoencoder(epochs=50):
    autoencoder = Autoencoder(latent_dim=512)
    discriminator = Discriminator()
    dataloader = get_default_dataloader()

    autoencoder, discriminator = train_gan_autoencoder_v2(autoencoder, discriminator, dataloader, epochs=epochs)

    # Save both
    #model_save_dir = my_filesystem.new_model_filepath('Autoencoders', False, '')
    #torch.save(autoencoder.state_dict(), my_filesystem.join_path(model_save_dir, 'autoencoder_gan6.pth'))
    #torch.save(discriminator.state_dict(), my_filesystem.join_path(model_save_dir, 'discriminator_gan6.pth'))

    model_save_dir = ['QWERT']
    g_path = my_filesystem.save_model(autoencoder, model_save_dir, 'autoencoder_gan.pth', True)
    d_path = my_filesystem.save_model(discriminator, model_save_dir, 'discriminator_gan.pth', False)

    load_and_test_model_from_path(g_path)