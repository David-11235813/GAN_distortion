import os
import numpy as np
import my_filesystem as files
from utilise_dataset import load_single_image_as_batch_of_1
from autoencoder_utils import Autoencoder, return_encoder_output_of_given_batch, return_decoder_output, load_autoencoder
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'


def check_all_partial_images_from_directory(autoencoder:Autoencoder, dir_name):
    autoencoder.eval()
    folder_path = files.join_path_unsupervised(files.dir_DATASET, dir_name)
    ranks = []
    for filename in os.listdir(folder_path):
        full_path = files.join_path_unsupervised(folder_path, filename)
        if not os.path.isfile(full_path): continue

        image = load_single_image_as_batch_of_1(full_path, is_grayscale=autoencoder.is_grayscale, resize_image_val=128).to(device)
        features = return_encoder_output_of_given_batch(autoencoder, image)

        # ! vvv those are all the latent_dim-images
        partial_reconstructions_tensor = return_decoder_output(autoencoder, features)

        #for each calculate it:
        print(full_path)
        print(partial_reconstructions_tensor.shape)
        r = return_rank_from_tensor(partial_reconstructions_tensor)
        ranks.append(r)
        print()

    ranks_array = np.array(ranks, dtype=np.float64)
    mean_rank = ranks_array.mean()
    std_rank = ranks_array.std()
    return ranks_array, mean_rank, std_rank


def return_rank_from_tensor(images):

    X = images.detach().cpu().numpy().astype(np.float64)
    X = X.reshape(X.shape[0], -1)

    rank = np.linalg.matrix_rank(X)
    mean = X.mean()
    std = X.std()

    print("n_images =", X.shape[0])
    print("n_features =", X.shape[1])
    print("rank =", rank)
    return rank



if __name__ == "__main__":
    load_path = files.get_model_path_from_directory(['AutoencodersTesting'], filename='autoencoder_gray_size16_100ep.pth')
    autoencoder = load_autoencoder(load_path)

    ranks_array1, mean_rank1, std_rank1 = check_all_partial_images_from_directory(autoencoder, 'faces_dataset_small_eval_00100')
    ranks_array2, mean_rank2, std_rank2 = check_all_partial_images_from_directory(autoencoder, 'faces_dataset_small_eval_01000')

    print(ranks_array1)
    print("Mean rank:", mean_rank1)
    print("Std rank:", std_rank1)

    print(ranks_array2)
    print("Mean rank:", mean_rank2)
    print("Std rank:", std_rank2)
    pass

