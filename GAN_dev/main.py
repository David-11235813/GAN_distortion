# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio


import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from testing import FlatImageFolder
from main2 import Autoencoder, train_autoencoder, show_reconstruction

device='cuda' if torch.cuda.is_available() else 'cpu'

def LoadModel():
    gen = Autoencoder().to(device)
    gen.load_state_dict(torch.load(os.path.join("autoencoder.pth"), map_location=device))
    gen.eval()
    return gen

def main():
    # Set up image preprocessing
    transform = transforms.Compose([
        transforms.Resize(128),         # resize the image (keeps values)
        transforms.CenterCrop(128),     # crop center region
        transforms.ToTensor(),          # converts to a float tensor and rescales pixels: 0..255 → 0.0..1.0; converts shape H×W×C → C×H×W.
        transforms.Normalize([0.5]*3,   # same as [0.5, 0.5, 0.5] ; normalizes (0,1) to (-1,1) for all channels
                             [0.5]*3)   # output_c = (input_c - mean[c]) / std[c]
    ])

    # Load LOCAL dataset - it works, but needs one more folder in file structure as label(s)
    #dataset = torchvision.datasets.ImageFolder(root='dataset/PhotosColorPicker', transform=transform)
    # this one doesn't need that label folder
    dataset = FlatImageFolder('..\\dataset\\faces_dataset_small', transform)

    #dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    '''
        for _ in range(5):
            images, _ = next(iter(dataloader))

            print(images.shape)
            grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
            plt.imshow(grid.permute(1,2,0))
            plt.axis("off")
            plt.show()
    '''

    # Create and train autoencoder
    model = Autoencoder(latent_dim=512)
    print("Training autoencoder...")
    model = train_autoencoder(model, dataloader, epochs=10)

    # Visualize results
    show_reconstruction(model, dataloader)

    # Save the trained model
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Model saved to autoencoder.pth")

def load_and_test_autoencoder():

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    dataset = FlatImageFolder('..\\dataset\\faces_dataset_small', transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    model = LoadModel()
    show_reconstruction(model, dataloader)


if __name__ == "__main__":
    #print(torch.cuda.is_available())
    #print(torch.version.cuda)
    #print(torch.backends.cudnn.enabled)

    main()
    #load_and_test_autoencoder()

