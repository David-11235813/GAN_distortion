import os
import time
import torch


dir_DEV = os.path.abspath(os.path.dirname(__file__))

dir_DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')
)

dir_DATASET_FACES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', 'faces_dataset_small')
)

dir_MID = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'middleman_folder')
)

dir_MID_MODELS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'middleman_folder', 'models')
)


def get_model_path_from_directory(directory: list[str], filename:str) -> str:
    model_path = os.path.join(dir_MID_MODELS, *directory, filename)
    return model_path


def prepare_model_save_path(directory: list[str], filename:str, new_timedate:bool) -> str:
    #if new_timedate and len(directory)>=1: directory[-1] = f"{time.strftime('%Y%m%d-%H%M%S')}_{directory[-1]}"
    if new_timedate: directory.append(time.strftime('%Y%m%d-%H%M%S'))
    save_dir = os.path.join(dir_MID_MODELS, *directory)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    return save_path