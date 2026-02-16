import os
from typing import List
from PIL import Image
import torch

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