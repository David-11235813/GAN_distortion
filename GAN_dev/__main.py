# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

import time
from __utilise_dataset import get_default_dataloader, default_dataset_path, test_dataloader
from __autoencoder import Autoencoder, train_autoencoder, show_autoencoder_reconstruction, load_and_test_model

import __my_filesystem as files


def main(epochs=50, dimensions=512, directory=['Autoencoders'], filename='autoencoder.pth'):
    # Set up image preprocessing: transformer and feeder of images
    dataloader = get_default_dataloader(default_dataset_path)
    #test_dataloader(dataloader, 5)

    # Create and train autoencoder
    autoencoder = Autoencoder(latent_dim=dimensions)
    print("Training autoencoder...")
    autoencoder = train_autoencoder(autoencoder, dataloader, epochs=epochs)

    # Visualize results
    show_autoencoder_reconstruction(autoencoder, dataloader)

    # Save the trained autoencoder
    save_path = files.save_model(autoencoder, directory, filename, False)
    print(f"Autoencoder saved to {save_path}")

    load_and_test_model(save_path, howmany=3)



if __name__ == "__main__":
    #check_cuda()   # w terminalu sprawdzanie czy GPU działa ok: nvidia-smi -l 1

    directory=['Autoencoders', 'LowDimension']
    main(100, 16, directory, 'autoencoder_16dim_100ep.pth') #'autoencoder_nogan_MSE_50ep.pth'

    #model_path = files.get_model_path_from_directory(['_presentable_models'],'autoencoder_nogan_MSE_50ep.pth')
    #load_and_test_model(model_path)

    #use_gan_autoencoder(10)
    #model_path = files.get_model_path_from_directory(['Autoencoders'],'autoencoder_gan5.pth')
    #load_and_test_model(model_path, 3) #it is in use_gan_autoencoder already

