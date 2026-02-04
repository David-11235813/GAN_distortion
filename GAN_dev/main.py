# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio


import torch

from utilise_dataset import get_default_dataloader, my_dataset_path, test_dataloader
from autoencoder import Autoencoder, train_autoencoder, show_reconstruction, LoadAutoencoder
from testing import *





def main():
    # Set up image preprocessing: transformer and feeder of images
    dataloader = get_default_dataloader(my_dataset_path)
    #test_dataloader(dataloader, 5)

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

    dataloader = get_default_dataloader(my_dataset_path)
    model = LoadAutoencoder()
    show_reconstruction(model, dataloader)


if __name__ == "__main__":
    #print(torch.cuda.is_available())
    #print(torch.version.cuda)
    #print(torch.backends.cudnn.enabled)

    main()
    #load_and_test_autoencoder()

