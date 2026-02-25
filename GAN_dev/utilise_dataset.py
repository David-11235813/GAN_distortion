import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import my_filesystem as files

default_dataset_path = files.dir_DATASET_FACES



# Load LOCAL dataset - code in next line works, but needs one more folder in file structure as label(s)
#dataset = torchvision.datasets.ImageFolder(root='dataset/PhotosColorPicker', transform=transform)
# this one doesn't need that label folder
#dataset = FlatImageFolder(my_dataset_path, transform)
class FlatImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return dummy label to stay compatible with DataLoader pipelines
        return image, 0


#double-check
def image_preprocessing_transform(resize_image_val:int=128):
    transform = transforms.Compose([
        transforms.Resize(resize_image_val),         # resize the image (keeps values)
        transforms.CenterCrop(resize_image_val),     # crop center region
        transforms.ToTensor(),          # converts to a float tensor and rescales pixels: 0..255 → 0.0..1.0; converts shape H×W×C → C×H×W.
        transforms.Normalize([0.5]*3,   # same as [0.5, 0.5, 0.5] ; normalizes (0,1) to (-1,1) for all channels
                             [0.5]*3)   # output_c = (input_c - mean[c]) / std[c]
    ])
    return transform


def image_preprocessing_transform_grayscale(resize_image_val:int=128):
    transform = transforms.Compose([
        transforms.Resize(resize_image_val),         # resize the image (keeps values)
        transforms.CenterCrop(resize_image_val),     # crop center region
        transforms.Grayscale(num_output_channels=1),   # <- convert to grayscale (PIL Image: 1 channel)
        transforms.ToTensor(),          # H×W×C → C×H×W (where C=1)
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform



def get_default_dataloader(path=default_dataset_path, is_grayscale=False, batch_size=32, resize_image_val=128):
    transform = image_preprocessing_transform(resize_image_val) if not is_grayscale else image_preprocessing_transform_grayscale(resize_image_val) #todo?
    dataset = FlatImageFolder(path, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,            # Faster GPU transfer
        persistent_workers=True,    # Keep workers alive between epochs (doubles speed); IMPORTANT True
        #prefetch_factor = 4         # Prefetch 4 batches per worker
    )
    dataloader.is_grayscale = is_grayscale
    dataloader.resize_image_val = resize_image_val
    return dataloader


def get_singular_dataloader(path=default_dataset_path, is_grayscale=False, resize_image_val=128):
    transform = image_preprocessing_transform(resize_image_val) if not is_grayscale else image_preprocessing_transform_grayscale(resize_image_val)
    dataset = FlatImageFolder(path, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,    # Keep workers alive between epochs (doubles speed); IMPORTANT True
    )
    dataloader.is_grayscale = is_grayscale
    dataloader.resize_image_val = resize_image_val
    return dataloader


def test_dataloader(dataloader: DataLoader, max_range : int=5):
    for _ in range(max_range):
        images, _ = next(iter(dataloader))

        print(images.shape)
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.axis("off")
        plt.show()


def load_single_image_as_batch_of_1(image_path, is_grayscale=False, resize_image_val=128):
    transform = image_preprocessing_transform(resize_image_val) if not is_grayscale else image_preprocessing_transform_grayscale(resize_image_val)
    img = Image.open(image_path).convert("L" if is_grayscale else "RGB")
    return transform(img).unsqueeze(0)  # add batch dim -> [1, C, H, W]