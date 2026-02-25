# pip install -r req_dev.txt
# python -m pip install --upgrade pip setuptools wheel
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
## if it doesn't work:
# python -m pip uninstall -y torch torchvision torchaudio
# python -m pip cache purge
# python -m pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

from utilise_dataset import get_default_dataloader, get_singular_dataloader
from autoencoder_utils import train_autoencoder, show_autoencoder_reconstruction, save_autoencoder, load_autoencoder, show_autoencoder_partial_reconstructions, show_autoencoder_partial_reconstructions_scaled
from GAN_feature_scales_trainer import train_gan_scales, save_GAN_scales, load_GAN_scales
import my_filesystem as files


_compression_size=16
_is_grayscale = True
_epochs=10
_directory=['AutoencodersTesting']
_filename=f'autoencoder{'_gray' if _is_grayscale else ''}_size{_compression_size}_{_epochs}ep.pth'
#_filename='autoencoder_gray_size8_100ep.pth'

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
        img_save_folder=files.get_default_img_folder_of_model(load_path)
    )

#-----------------------------------------------------------------------------------------------------------------------
def main1():
    #autoencoder_path = files.get_model_path_from_directory(directory=_directory, filename=_filename)
    #autoencoder_path = produce_autoencoder(compression_size=16, is_grayscale=True, epochs=10, directory=_directory, filename='autoencoder_gray_size16_10ep.pth') #uncomment for training new autoencoder; leave commented for loading previous one

    autoencoder_path = produce_autoencoder() #preset parameters
    use_autoencoder(autoencoder_path, 2)


#singular features
def main2():
    autoencoder_path = files.get_model_path_from_directory(_directory, _filename)
    autoencoder = load_autoencoder(autoencoder_path)

    show_autoencoder_partial_reconstructions(autoencoder, howmany_plots=5, save=True,
        img_save_folder=files.prepare_folder(files.join_path_unsupervised(files.dir_MID, 'IMG_TEST'))
    )

def main3():
    autoencoder_path = files.get_model_path_from_directory(_directory, _filename)
    autoencoder = load_autoencoder(autoencoder_path)
    dataloader = get_default_dataloader(files.dir_DATASET_FACES, autoencoder.is_grayscale)

    scales = train_gan_scales(autoencoder, dataloader, epochs=10)
    # todo: save the scales in a file

    directory = _directory
    filename = _filename

    save_path = save_GAN_scales(scales,
        scales_save_folder=files.get_default_img_folder_of_model(autoencoder_path),
        scales_filename=''
    )
    print(f"Scales were saved to {save_path}")

    print(load_GAN_scales(save_path))



if __name__ == "__main__":
    #main1()     # generate and test the autoencoder, show results (and optionally save)
    #main2()     # show autoencoder's partial reconstructions
    main3()     # train GAN-autoencoder (no saving)
    #todo:
    # incorporate decoder-scaling into autoencoder's structure (easy qol)
    # fix paths and folders


    pass

