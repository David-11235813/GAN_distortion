import matplotlib.pyplot as plt


# Show original vs reconstruction
def display_autoencoder_reconstructions(images_batch, reconstructions, howmany):
    howmany = min(howmany, images_batch.shape[0])

    fig, axes = plt.subplots(2, howmany, figsize=(15, 4))
    for i in range(howmany):
        # Original
        orig = images_batch[i].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2  # Denormalize
        #axes[0, i].imshow(orig)    # works only for RGB, not grayscale
        axes[0, i].imshow(orig.squeeze(-1) if orig.shape[2] == 1 else orig,
                          cmap='gray' if orig.shape[2] == 1 else None, vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstruction
        recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        recon = (recon + 1) / 2  # Denormalize
        #axes[1, i].imshow(recon)    # works only for RGB, not grayscale
        axes[1, i].imshow(recon.squeeze(-1) if recon.shape[2] == 1 else recon,
                          cmap='gray' if recon.shape[2] == 1 else None, vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.show()




def reconstruct_image_and_display_features(autoencoder, image1):
    return
    '''
    reconstruction, features = autoencoder(image1)

    nr_features = autoencoder.latent_dim
    image = images[0].cpu().permute(1, 2, 0).numpy()

    # Show original vs reconstruction
    fig, axes = plt.subplots(2, nr_features, figsize=(15, 4))

    # Original
    orig = images[0].cpu().permute(1, 2, 0).numpy()
    orig = (orig + 1) / 2  # Denormalize
    axes[0, 0].imshow(orig)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original | Reconstructed', fontsize=10)

    # Reconstructed
    recon = reconstructions[0].cpu().permute(1, 2, 0).numpy()
    recon = (recon + 1) / 2  # Denormalize
    axes[0, 1].imshow(recon)
    axes[0, 1].axis('off')

    # Every part of reconstruction
    for i in range(nr_features):

        # Reconstruction
        part = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        part = (part + 1) / 2  # Denormalize
        axes[1, i].imshow(part)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Parts', fontsize=10)

    plt.tight_layout()
    plt.show()
    '''


def save_plot(plot, path):
    pass