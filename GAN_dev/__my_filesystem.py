import os
import time
import torch

# get all the directories
# macro them here
#
# then get all functions to save and read from those directories


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

def dir_TEMP():
    temp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'temp')
    )
    os.makedirs(temp, exist_ok=True)
    return temp


def goto_subdirectory(directory, subdirectory):
    subdirectory_path = os.path.join(directory, subdirectory)
    os.makedirs(directory, exist_ok=True)
    return subdirectory_path


def join_path(*parts: str) -> str:
    return os.path.join(*parts)


def new_model_filepath(subdirectory:str=None, timedate:bool=True, details:str='') -> str:

    #default behaviour (original)
    model_path = os.path.join(
        dir_MID_MODELS,
        time.strftime("%Y%m%d-%H%M%S")
    )

    #otherwise, add subdirectory or remove timedate
    if (subdirectory is not None) or (timedate is False) or (details is not None):
        if details != '': details = str(f"__{details}")
        model_path = os.path.join(
            dir_MID_MODELS,
            *([subdirectory] if subdirectory else []),
            *([time.strftime(f"%Y%m%d-%H%M%S{details}")] if timedate else [*([details] if details != '' else [])])
        )

    os.makedirs(model_path, exist_ok=True)
    return model_path


def goto_model(*parts_of_path: str) -> str:
    model_path = os.path.join(
        dir_MID_MODELS,
        *([*parts_of_path] if parts_of_path else [])
    )
    return model_path




#prepare_model_savepath
def save_model(model:torch.nn.Module, directory: list[str], filename:str, new_timedate:bool) -> str:
    #if new_timedate and len(directory)>=1: directory[-1] = f"{time.strftime('%Y%m%d-%H%M%S')}_{directory[-1]}"
    if new_timedate: directory.append(time.strftime('%Y%m%d-%H%M%S'))
    save_dir = os.path.join(dir_MID_MODELS, *directory)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'latent_dim': model.latent_dim
    }, save_path)
    return save_path



def get_model_path_from_directory(directory: list[str], filename:str) -> str:
    model_path = os.path.join(dir_MID_MODELS, *directory, filename)
    return model_path
