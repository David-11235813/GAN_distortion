import matplotlib.pyplot as plt
import math


# Show original vs reconstruction
def display_autoencoder_reconstructions(images_batch, reconstructions, howmany_imgs) -> plt.Figure:
    howmany = min(howmany_imgs, images_batch.shape[0])
    is_grayscale:bool = (images_batch[0].cpu().permute(1, 2, 0).numpy().shape[2] == 1)

    fig, axes = plt.subplots(2, howmany, figsize=(15, 4))
    axes[0, 0].set_title('Original', fontsize=10)
    axes[1, 0].set_title('Reconstructed', fontsize=10)
    [ax.axis('off') for ax in axes.ravel()] #disables all axes

    for i in range(howmany):
        # Original
        orig = images_batch[i].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2  # Denormalize
        axes[0, i].imshow(orig.squeeze(-1) if is_grayscale else orig,
                          cmap='gray' if is_grayscale else None, vmin=0, vmax=1)

        # Reconstruction
        recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        recon = (recon + 1) / 2  # Denormalize
        axes[1, i].imshow(recon.squeeze(-1) if is_grayscale else recon,
                          cmap='gray' if is_grayscale else None, vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()
    return fig


def display_autoencoder_reconstructions_OneImg(original, full_reconstruction, partial_reconstructions_tensor) -> plt.Figure:
    howmany = partial_reconstructions_tensor.shape[0]
    is_grayscale:bool = (original.cpu().permute(1, 2, 0).numpy().shape[2] == 1)
    cols = 8
    partial_rows = math.ceil(howmany / cols) if howmany > 0 else 0
    rows = 1 + partial_rows
    # choose per-image size (in inches). Tweak these to taste.
    per_image_w = 2.0   # width per subplot
    per_image_h = 2.0   # height per subplot

    fig_w = cols * per_image_w
    fig_h = rows * per_image_h
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    #fig, axes = plt.subplots(1+(howmany//8), 8, figsize=(15, 4))

    #[ax.axis('off') for ax in axes.ravel()] #disables all axes
    [ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) for ax in axes.ravel()] #preserves borders only
    [ax.axis('off') for ax in axes[0, 2:]]

    axes[0, 0].set_title('Original', fontsize=10)
    # Original
    orig = original.cpu().permute(1, 2, 0).numpy()
    orig = (orig + 1) / 2  # Denormalize
    axes[0, 0].imshow(orig.squeeze(-1) if is_grayscale else orig,
                      cmap='gray' if is_grayscale else None, vmin=0, vmax=1)


    axes[0, 1].set_title('Reconstructed', fontsize=10)
    # Reconstruction
    recon = full_reconstruction.cpu().permute(1, 2, 0).numpy()
    recon = (recon + 1) / 2  # Denormalize
    axes[0, 1].imshow(recon.squeeze(-1) if is_grayscale else recon,
                      cmap='gray' if is_grayscale else None, vmin=0, vmax=1)

    def print_partials():
        for j in range(partial_rows):
            for i in range(cols):
                if j*cols+i >= howmany: return
                axes[1+j, i].set_title(f'feature {cols*j+i+1}', fontsize=10)
                # Reconstructions
                recon = partial_reconstructions_tensor[cols*j+i].cpu().permute(1, 2, 0).numpy()
                recon = (recon + 1) / 2  # Denormalize
                axes[1+j, i].imshow(recon.squeeze(-1) if is_grayscale else recon,
                              cmap='gray' if is_grayscale else None, vmin=0, vmax=1)
        return

    print_partials()


    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.12, hspace=0.25, left=0.03, right=0.97, top=0.95, bottom=0.03)
    plt.show()
    return fig






