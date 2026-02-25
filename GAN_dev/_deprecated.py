from typing import List
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder_utils import Autoencoder
from utilise_dataset import get_default_dataloader
from my_filesystem import *
from autoencoder_utils import Autoencoder, device, files


def log(text):
    print(text)

def list_jpg_files(imagesPath: os.path) -> List[str]:
    return sorted(f for f in os.listdir(imagesPath)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(imagesPath, f)))

#open f files from arbitrary folder in default photos application
def access_dataset():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'dataset'))
    img_folder_path = os.path.join(dataset_path, 'PhotosColorPicker')
    log(img_folder_path)

    images = list_jpg_files(img_folder_path)
    log(images)

    f = 1
    for image in images:
        jpg = Image.open(os.path.join(img_folder_path, image))
        jpg.show()

        if f >= 3: break
        f+=1



if __name__ == '__main__':
    #access_dataset()
    #testing()
    pass





#pip install -r req_dev.txt


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# This function will help us visualize images later
# It's crucial to be able to see what your data looks like
def show_images(images, title="Images"):
    """
    Display a batch of images in a grid.

    Args:
        images: Tensor of images with shape (batch, channels, height, width)
        title: Title for the plot
    """
    # We need to move images to CPU and convert to numpy for matplotlib
    images = images.cpu().numpy()

    # Determine grid size based on batch size
    batch_size = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(batch_size)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if batch_size > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < batch_size:
            # Images are in format (channels, height, width)
            # We need to transpose to (height, width, channels) for display
            img = np.transpose(images[idx], (1, 2, 0))

            # If the image was normalized to [-1, 1], we need to rescale to [0, 1]
            # for proper display
            img = (img + 1) / 2  # This converts from [-1,1] to [0,1]
            img = np.clip(img, 0, 1)  # Make sure we're in valid range

            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'dataset'))
img_folder_path = os.path.join(dataset_path, 'PhotosColorPicker')

def create_data_pipeline(data_path=img_folder_path, image_size=128, batch_size=16):
    """
    Create a complete data pipeline for loading and preprocessing images.

    This function encapsulates all the decisions about how to prepare your data.
    Let me explain each choice:

    Args:
        data_path: Where to download/find the dataset
        image_size: What size to resize images to (128x128 is a good starting point)
        batch_size: How many images to process at once
    """

    # Define the transformations we'll apply to each image
    # Think of this as a recipe that every image follows
    transform = transforms.Compose([
        # First, resize all images to a consistent size
        # Why? Because neural networks need consistent input dimensions
        # 128x128 is large enough to preserve detail but small enough to train quickly
        transforms.Resize(image_size),

        # Center crop to ensure we have exactly the size we want
        # This handles images that aren't perfectly square
        transforms.CenterCrop(image_size),

        # Convert PIL Image to PyTorch tensor
        # This changes the data from a PIL image object to a numerical array
        transforms.ToTensor(),

        # Normalize pixel values from [0, 1] to [-1, 1]
        # Why [-1, 1]? This range often works better for neural networks
        # The values (0.5, 0.5, 0.5) are the mean and std for each channel
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # For this example, I'm using CIFAR-10 because it's easy to download
    # In your actual project, you'll want to replace this with CelebA or FFHQ
    # I'll show you how to do that after this example
    dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    # Create a DataLoader that will feed batches to your network
    # The DataLoader handles shuffling, batching, and parallel loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the data each epoch for better training
        num_workers=2,  # Use multiple processes to load data faster
        pin_memory=True  # Optimization for GPU training
    )

    return dataloader, dataset


def test_data_pipeline():
    """
    This function tests our pipeline to make sure everything works.
    Always test your components before building on top of them!
    """
    print("Creating data pipeline...")
    dataloader, dataset = create_data_pipeline(batch_size=16)

    print(f"Dataset contains {len(dataset)} images")
    print(f"Each batch will contain {dataloader.batch_size} images")

    # Get one batch of data
    images, labels = next(iter(dataloader))

    print(f"Batch shape: {images.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")

    # Display the batch
    show_images(images, title="Sample batch from dataset")

    # Let's also look at a single image in detail
    single_image = images[0]
    print(f"\nSingle image shape: {single_image.shape}")
    print(f"This means: {single_image.shape[0]} channels, "
          f"{single_image.shape[1]}x{single_image.shape[2]} pixels")


if __name__ == "__main__":
    # Run the test
    test_data_pipeline()

#--------------------------------------------------------------------------------------------------------

#pip install -r req_dev.txt


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# This function will help us visualize images later
# It's crucial to be able to see what your data looks like
def show_images(images, title="Images"):
    """
    Display a batch of images in a grid.

    Args:
        images: Tensor of images with shape (batch, channels, height, width)
        title: Title for the plot
    """
    # We need to move images to CPU and convert to numpy for matplotlib
    images = images.cpu().numpy()

    # Determine grid size based on batch size
    batch_size = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(batch_size)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if batch_size > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < batch_size:
            # Images are in format (channels, height, width)
            # We need to transpose to (height, width, channels) for display
            img = np.transpose(images[idx], (1, 2, 0))

            # If the image was normalized to [-1, 1], we need to rescale to [0, 1]
            # for proper display
            img = (img + 1) / 2  # This converts from [-1,1] to [0,1]
            img = np.clip(img, 0, 1)  # Make sure we're in valid range

            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'dataset'))
img_folder_path = os.path.join(dataset_path, 'PhotosColorPicker')

def create_data_pipeline(data_path=img_folder_path, image_size=128, batch_size=16):
    """
    Create a complete data pipeline for loading and preprocessing images.

    This function encapsulates all the decisions about how to prepare your data.
    Let me explain each choice:

    Args:
        data_path: Where to download/find the dataset
        image_size: What size to resize images to (128x128 is a good starting point)
        batch_size: How many images to process at once
    """

    # Define the transformations we'll apply to each image
    # Think of this as a recipe that every image follows
    transform = transforms.Compose([
        # First, resize all images to a consistent size
        # Why? Because neural networks need consistent input dimensions
        # 128x128 is large enough to preserve detail but small enough to train quickly
        transforms.Resize(image_size),

        # Center crop to ensure we have exactly the size we want
        # This handles images that aren't perfectly square
        transforms.CenterCrop(image_size),

        # Convert PIL Image to PyTorch tensor
        # This changes the data from a PIL image object to a numerical array
        transforms.ToTensor(),

        # Normalize pixel values from [0, 1] to [-1, 1]
        # Why [-1, 1]? This range often works better for neural networks
        # The values (0.5, 0.5, 0.5) are the mean and std for each channel
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # For this example, I'm using CIFAR-10 because it's easy to download
    # In your actual project, you'll want to replace this with CelebA or FFHQ
    # I'll show you how to do that after this example
    dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    # Create a DataLoader that will feed batches to your network
    # The DataLoader handles shuffling, batching, and parallel loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the data each epoch for better training
        num_workers=2,  # Use multiple processes to load data faster
        pin_memory=True  # Optimization for GPU training
    )

    return dataloader, dataset


def test_data_pipeline():
    """
    This function tests our pipeline to make sure everything works.
    Always test your components before building on top of them!
    """
    print("Creating data pipeline...")
    dataloader, dataset = create_data_pipeline(batch_size=16)

    print(f"Dataset contains {len(dataset)} images")
    print(f"Each batch will contain {dataloader.batch_size} images")

    # Get one batch of data
    images, labels = next(iter(dataloader))

    print(f"Batch shape: {images.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")

    # Display the batch
    show_images(images, title="Sample batch from dataset")

    # Let's also look at a single image in detail
    single_image = images[0]
    print(f"\nSingle image shape: {single_image.shape}")
    print(f"This means: {single_image.shape[0]} channels, "
          f"{single_image.shape[1]}x{single_image.shape[2]} pixels")


if __name__ == "__main__":
    # Run the test
    test_data_pipeline()

def timing():
    import time
    start = time.time()  # now do stuff
    end = time.time()  # finished doing stuff
    print(f"Total runtime of the program is {end - start} seconds")

def check_cuda(test : bool=False):
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.backends.cudnn.enabled)
    if(torch.cuda.is_available() and test):
        cuda = torch.device('cuda')  # Default CUDA device
        cuda0 = torch.device('cuda:0')
        cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

        x = torch.tensor([1., 2.], device=cuda0)
        # x.device is device(type='cuda', index=0)
        y = torch.tensor([1., 2.]).cuda()
        # y.device is device(type='cuda', index=0)

        with torch.cuda.device(1):
            # allocates a tensor on GPU 1
            a = torch.tensor([1., 2.], device=cuda)

            # transfers a tensor from CPU to GPU 1
            b = torch.tensor([1., 2.]).cuda()
            # a.device and b.device are device(type='cuda', index=1)

            # You can also use ``Tensor.to`` to transfer a tensor:
            b2 = torch.tensor([1., 2.]).to(device=cuda)
            # b.device and b2.device are device(type='cuda', index=1)

            c = a + b
            # c.device is device(type='cuda', index=1)

            z = x + y
            # z.device is device(type='cuda', index=0)

            # even within a context, you can specify the device
            # (or give a GPU index to the .cuda call)
            d = torch.randn(2, device=cuda2)
            e = torch.randn(2).to(cuda2)
            f = torch.randn(2).cuda(cuda2)
            # d.device, e.device, and f.device are all device(type='cuda', index=2)

def test_cuda(is_cpu : bool=False):
    import time
    start = time.time()  # now do stuff

    my_device = None
    if(is_cpu): my_device = 'cpu'
    else: my_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a large tensor on GPU
    x = torch.randn(10000, 10000, device=my_device)
    y = torch.randn(10000, 10000, device=my_device)

    # Do a computation - this should spike GPU usage
    z = torch.matmul(x, y)

    print(f"Computation done on: {z.device}")
    print("Check nvidia-smi now - you should see memory usage and brief GPU spike")

    end = time.time()  # finished doing stuff
    print(f"Total runtime of the program is {end - start} seconds")





#deprecated
def show_autoencoder_reconstruction_TESTING(autoencoder:Autoencoder, dataloader, howmany_plots=1, save:bool=False, img_save_folder:str= ''):
    autoencoder.eval()
    with torch.no_grad():
        for idx in range(howmany_plots):
            images, _ = next(iter(dataloader))
            images = images.to(device)
            reconstructions, features = autoencoder(images)

            fig = display_autoencoder_reconstructions_TESTING(images, reconstructions, 8)

            if save: fig.savefig(files.join_path_unsupervised(img_save_folder, f'features_{idx}.png'))
            print(f"Feature vector size: {features.shape}")



#deprecated
def display_autoencoder_reconstructions_TESTING(images_batch, reconstructions, howmany_imgs) -> plt.Figure:
    howmany = min(howmany_imgs, images_batch.shape[0])
    is_grayscale:bool = (images_batch[0].cpu().permute(1, 2, 0).numpy().shape[2] == 1)

    fig, axes = plt.subplots(2, howmany, figsize=(15, 4))
    #[ax.axis('off') for ax in axes.ravel()] #disables all axes
    [ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) for ax in axes.ravel()] #preserves borders only

    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 1].set_title('Reconstructed', fontsize=10)
    axes[1, 0].set_title('1st feature', fontsize=10)

    for i in range(howmany):
        # Original
        orig = images_batch[i].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2  # Denormalize
        axes[0, i].imshow(orig)    # works only for RGB, not grayscale

        if i>0: continue
        # Reconstruction
        recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        recon = (recon + 1) / 2  # Denormalize
        axes[1, i].imshow(recon)    # works only for RGB, not grayscale

    plt.tight_layout()
    plt.show()
    return fig



#pyplot testing
'''
def main3():
    load_path = files.get_model_path_from_directory(directory=_directory, filename=_filename)
    autoencoder = load_autoencoder(load_path)
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, autoencoder.is_grayscale)

    show_autoencoder_reconstruction_TESTING(autoencoder, dataloader, howmany_plots=2, save=True,
        img_save_folder=files.prepare_folder(files.join_path_unsupervised(files.dir_MID, 'IMG_TEST'))
    )
'''







#temporary checks and proofs of concept here.



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


'''
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
    g_path = files.save_model(autoencoder, model_save_dir, 'autoencoder_gan.pth', True)
    d_path = files.save_model(discriminator, model_save_dir, 'discriminator_gan.pth', False)

    load_and_test_model(g_path)
'''



def check_cuda():
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.backends.cudnn.enabled)

    #temp
    start = time.time()  # now do stuff

    end = time.time()  # finished doing stuff
    print(f"Total runtime of the program is {end - start} seconds")

def dir_TEMP():
    temp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'temp')
    )
    os.makedirs(temp, exist_ok=True)
    return temp

def goto_subdirectory(directory, subdirectory):
    subdirectory_path = os.path.join(directory, subdirectory)
    os.makedirs(directory, exist_ok=True)
    return subdirectory_path

def new_model_filepath(subdirectory: str = None, timedate: bool = True, details: str = '') -> str:

    # default behaviour (original)
    model_path = os.path.join(
        dir_MID_MODELS,
        time.strftime("%Y%m%d-%H%M%S")
    )

    # otherwise, add subdirectory or remove timedate
    if (subdirectory is not None) or (timedate is False) or (details is not None):
        if details != '': details = str(f"__{details}")
        model_path = os.path.join(
            dir_MID_MODELS,
            *([subdirectory] if subdirectory else []),
            *([time.strftime(f"%Y%m%d-%H%M%S{details}")] if timedate else [*([details] if details != '' else [])])
        )

    os.makedirs(model_path, exist_ok=True)
    return model_path

