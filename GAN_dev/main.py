# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
## if it doesn't work:
# python -m pip uninstall -y torch torchvision torchaudio
# python -m pip cache purge
# python -m pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

from utilise_dataset import get_default_dataloader, get_singular_dataloader
from autoencoder_utils import train_autoencoder, show_autoencoder_reconstruction, save_autoencoder, load_autoencoder, show_autoencoder_partial_reconstructions
import my_filesystem as files


_compression_size=12
_is_grayscale = True
_epochs=50
_directory=['AutoencodersTesting']
_filename=f'autoencoder{'_gray' if _is_grayscale else ''}_size{_compression_size}_{_epochs}ep.pth'
#_filename='autoencoder_size32_50ep.pth'

def produce_autoencoder(compression_size=_compression_size, is_grayscale=_is_grayscale, epochs=_epochs, directory=_directory, filename=_filename) -> str:
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, is_grayscale)
    autoencoder = train_autoencoder(dataloader, compression_size, epochs)
    show_autoencoder_reconstruction(autoencoder, dataloader)
    save_path = save_autoencoder(autoencoder, directory, filename, False)
    print(f"Autoencoder saved to {save_path}")
    return save_path


def use_autoencoder(load_path:str, howmany_plots=1):
    autoencoder = load_autoencoder(load_path)
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, autoencoder.is_grayscale)

    show_autoencoder_reconstruction(autoencoder, dataloader, howmany_plots,
        save=True,
        img_save_folder=files.get_default_img_folder(load_path)
    )

#-----------------------------------------------------------------------------------------------------------------------
def main1():
    """
    generate and test the autoencoder, show results (and optionally save)
    """
    #autoencoder_path = files.get_model_path_from_directory(directory=_directory, filename=_filename)
    #autoencoder_path = produce_autoencoder(compression_size=16, is_grayscale=True, epochs=10, directory=_directory, filename='autoencoder_gray_size16_10ep.pth') #uncomment for training new autoencoder; leave commented for loading previous one

    autoencoder_path = produce_autoencoder() #preset parameters
    use_autoencoder(autoencoder_path, 5)


#singular features
def main2():
    autoencoder_path = files.get_model_path_from_directory(_directory, _filename)
    autoencoder = load_autoencoder(autoencoder_path)

    show_autoencoder_partial_reconstructions(autoencoder, howmany_plots=5, save=True,
        img_save_folder=files.prepare_folder(files.join_path_unsupervised(files.dir_MID, 'IMG_TEST'))
    )



if __name__ == "__main__":
    #main1()
    main2()

    #todo:
    # show reproduced features


    pass

