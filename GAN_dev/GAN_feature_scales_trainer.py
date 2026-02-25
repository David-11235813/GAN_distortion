import torch
import torch.nn as nn
import my_filesystem as files
from autoencoder_utils import Autoencoder
from GAN_scales_rules import combine_by_rule, first_scale_is_105proc

device="cuda" if torch.cuda.is_available() else "cpu"

class Discriminator(nn.Module):
    def __init__(self, img_side_pixels=128, channels_nr=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels_nr, img_side_pixels, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(img_side_pixels, img_side_pixels * 2, 4, 2, 1),
            nn.BatchNorm2d(img_side_pixels * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(img_side_pixels * 2, img_side_pixels * 4, 4, 2, 1),
            nn.BatchNorm2d(img_side_pixels * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(img_side_pixels * 4, img_side_pixels * 8, 4, 2, 1),
            nn.BatchNorm2d(img_side_pixels * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(img_side_pixels * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1),  # forces output to (batch, 1, 1, 1) regardless of input size
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)


# ── Training Function ────────────────────────────────────────────────────────

def train_gan_scales(
    generator : Autoencoder,
    #discriminator : Discriminator,
    dataloader,
    epochs=50,
    #lr=0.0002,
    #device=device,
):
    generator.to(device)

    discriminator = Discriminator(dataloader.resize_image_val, channels_nr = 3 if not generator.is_grayscale else 1)
    discriminator.to(device)


    rule_map = first_scale_is_105proc()
    # make sure rule makes sense
    if len(rule_map) > generator.latent_dim - 1: # at least one trainable scale
        raise ValueError(f"len(rule)={len(rule_map)} exceeds or equals latent_dim={generator.latent_dim}. Max amount is latent_dim-1.")
    if any((p < 0 or p >= generator.latent_dim) for p in set(rule_map.keys())):
        raise ValueError("rule position out of range for resulting length")

    trained_scales = torch.ones(generator.latent_dim - len(rule_map), requires_grad=True, device=device)


    modified_loss_formula = False
    #product_loss = (trained_scales.prod() - pow(1.03, generator.latent_dim)).pow(2)
    desired_geometric_average = pow(1.03, generator.latent_dim)
    # og value: 0.1
    lambda_prod = 0.05 # how much encouragement does the model need to derive all scales from 1.0 by approx. 3%
    # og value: 0.01; 0.05 is too much, try 0.02
    lambda_div = 0.02 # how much encouragement does the model need to increase the variance of scales




    # generator.weights is its only trainable parameter
    opt_g = torch.optim.Adam([trained_scales], lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)


            # Can add label smoothing, for the Discriminator not to win completely
            #real_labels = torch.full((batch_size, 1), 0.9, device=device)  # instead of 1.0
            #fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # instead of 0.0
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ── Train Discriminator ──────────────────────────────────────────

            #encoded = return_encoder_output_of_given_batch(generator, real_imgs)
            #decoded = return_decoder_output(generator, encoded * trained_scales)
            #fake_imgs = decoded.detach()

            #fake_imgs = generator(batch_size*real_imgs[])#.detach()  # detach: no G update here

            with torch.no_grad():
                encoded = generator.encode(real_imgs)

            decoded_og = generator.decode(encoded).detach()
            #fake_imgs = generator.decode(encoded, trained_scales).detach()  # detached for D
            full_scales = combine_by_rule(trained_scales, rule_map)
            fake_imgs = generator.decode(encoded, full_scales).detach()  # CHANGED - we only train trainable scales, but reconstruct with full

            d_loss = (
                criterion(discriminator(decoded_og), real_labels) + # prevents decoder to learn decoding artifacts and overpowering learning
                criterion(discriminator(fake_imgs), fake_labels)
            )

            # Can update D every other step - so that discriminator doesn't win completely
            #if batch_idx % 2 == 0:
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ── Train Generator ──────────────────────────────────────────────
            with torch.no_grad():
                encoded = generator.encode(real_imgs)  # encoder doesn't need grads
            #fake_imgs_g = generator.decode(encoded, trained_scales)  # we supply full scales to decoder
            full_scales = combine_by_rule(trained_scales, rule_map)
            fake_imgs_g = generator.decode(encoded, full_scales)  # no detach — grads flow to trained_scales

            g_loss = criterion(discriminator(fake_imgs_g), real_labels)
            if modified_loss_formula:
                product_loss = (trained_scales.prod() - desired_geometric_average).pow(2)
                diversity_loss = -trained_scales.var()  # negative variance = penalize low spread
                g_loss = g_loss + lambda_prod * product_loss + lambda_div * diversity_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")
        print(f'trained scales (size={trained_scales.size()}):\n{trained_scales}')
        full_scales = combine_by_rule(trained_scales, rule_map)
        print(f'full scales: (size={full_scales.size()}):\n{full_scales}')
        print()

    full_scales = combine_by_rule(trained_scales, rule_map)
    return full_scales


def save_GAN_scales(scales:torch.Tensor, scales_save_folder:str, scales_filename:str=None) -> str:
    if scales_filename is None or scales_filename == '':
        scales_filename = 'scales_untitled.pth'
    name, ext = files.split_ext(scales_filename)
    scales_filename = f'{name}_{files.get_datetime_str()}_{ext}'

    scales_save_path = files.join_path_unsupervised(scales_save_folder, scales_filename)

    torch.save(scales, scales_save_path)
    return scales_save_path


def load_GAN_scales(path: str) -> torch.Tensor:
    return torch.load(path, map_location=device)