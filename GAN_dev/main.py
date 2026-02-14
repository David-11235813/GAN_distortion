# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

from __utilise_dataset import get_default_dataloader, get_singular_dataloader

from autoencoder_utils import train_autoencoder, show_autoencoder_reconstruction, save_autoencoder, load_autoencoder, show_encoder_output
import my_filesystem as files


_compression_size=32
_is_grayscale = True
_epochs=5
_directory=['AutoencodersTesting']
_filename='autoencoder1.pth'


def produce_autoencoder(compression_size=_compression_size, is_grayscale=_is_grayscale, epochs=_epochs, directory=_directory, filename=_filename) -> str:
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, is_grayscale)
    autoencoder = train_autoencoder(dataloader, compression_size, epochs)
    show_autoencoder_reconstruction(autoencoder, dataloader)
    save_path = save_autoencoder(autoencoder, directory, filename, False)
    print(f"Autoencoder saved to {save_path}")
    return save_path


def use_autoencoder(load_path:str, howmany=1):
    autoencoder = load_autoencoder(load_path)
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, autoencoder.is_grayscale)
    for _ in range(howmany):
        show_autoencoder_reconstruction(autoencoder, dataloader)

#-----------------------------------------------------------------------------------------------------------------------

def main1():
    #autoencoder_path = files.get_model_path_from_directory(directory=_directory, filename='autoencoder_size16_100ep.pth')
    autoencoder_path = produce_autoencoder(compression_size=16, is_grayscale=True, epochs=10, directory=_directory, filename='autoencoder_gray_size16_10ep.pth') #uncomment for training new autoencoder; leave commented for loading previous one
    use_autoencoder(autoencoder_path, 3)


if __name__ == "__main__":

    main1()

    #todo:
    # finish grayscale implementation
    # then kartka.

    '''
    autoencoder_path = files.get_model_path_from_directory(_directory, _filename)

    # now: display only the _compression_size numbers
    dataloader = get_singular_dataloader(files.dir_DATASET_FACES)
    autoencoder = load_autoencoder(autoencoder_path)

    features = show_encoder_output(autoencoder, dataloader)
    print(features[0].tolist()) 21:09
    '''

