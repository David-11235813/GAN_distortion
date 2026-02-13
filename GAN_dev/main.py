# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

from __utilise_dataset import get_default_dataloader

from autoencoder_utils import train_autoencoder, show_autoencoder_reconstruction, save_autoencoder, load_autoencoder
import my_filesystem as files


_compression_size=32
_epochs=5
_directory=['AutoencodersTesting']
_filename='autoencoder1.pth'


def produce_autoencoder(compression_size=_compression_size, epochs=_epochs, directory=_directory, filename=_filename) -> str:
    dataloader = get_default_dataloader(files.dir_DATASET_FACES)
    print("Training autoencoder...")
    autoencoder = train_autoencoder(dataloader, compression_size, epochs)
    show_autoencoder_reconstruction(autoencoder, dataloader)
    save_path = save_autoencoder(autoencoder, directory, filename, False)
    print(f"Autoencoder saved to {save_path}")
    return save_path


def use_autoencoder(load_path:str, howmany=1):
    dataloader = get_default_dataloader(files.dir_DATASET_FACES)
    autoencoder = load_autoencoder(load_path)
    for _ in range(howmany):
        show_autoencoder_reconstruction(autoencoder, dataloader)


if __name__ == "__main__":
    autoencoder_path = files.get_model_path_from_directory(_directory, _filename)
    #autoencoder_path = produce_autoencoder() #uncomment for training new autoencoder; leave commented for loading previous one

    use_autoencoder(autoencoder_path, 3)