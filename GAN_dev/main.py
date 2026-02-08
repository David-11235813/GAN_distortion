# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

import time
from utilise_dataset import get_default_dataloader, default_dataset_path, test_dataloader
from autoencoder import Autoencoder, train_autoencoder, show_reconstruction, load_and_test_model_from_path, load_and_test_model_from_directory
from testing import *
import my_filesystem


def main(epochs=50):
    # Set up image preprocessing: transformer and feeder of images
    dataloader = get_default_dataloader(default_dataset_path)
    #test_dataloader(dataloader, 5)

    # Create and train autoencoder
    autoencoder = Autoencoder(latent_dim=512)
    print("Training autoencoder...")
    autoencoder = train_autoencoder(autoencoder, dataloader, epochs=epochs)

    # Visualize results
    show_reconstruction(autoencoder, dataloader)

    # Save the trained autoencoder
    save_path = my_filesystem.save_model(autoencoder, ['Autoencoders'], 'autoencoder_nogan_MSE_50ep.pth', True)
    print(f"Autoencoder saved to {save_path}")

    load_and_test_model_from_path(save_path, howmany=3)




def check_cuda():
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.backends.cudnn.enabled)

    #temp
    start = time.time()  # now do stuff

    end = time.time()  # finished doing stuff
    print(f"Total runtime of the program is {end - start} seconds")


if __name__ == "__main__":
    #check_cuda()   # w terminalu sprawdzanie czy GPU działa ok: nvidia-smi -l 1

    #main(10)
    #load_and_test_model_from_directory(['_presentable_models'], 'autoencoder_nogan_MSE_50ep.pth')

    use_gan_autoencoder(10)
    #load_and_test_model_from_directory('autoencoder_gan_best.pth', 10) #it is in use_gan_autoencoder already

