import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import __my_filesystem as files

default_dataset_path = files.dir_DATASET_FACES



# Load LOCAL dataset - it works, but needs one more folder in file structure as label(s)
# dataset = torchvision.datasets.ImageFolder(root='dataset/PhotosColorPicker', transform=transform)
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


# todo
def image_preprocessing_transform():
    transform = transforms.Compose([
        transforms.Resize(128),         # resize the image (keeps values)
        transforms.CenterCrop(128),     # crop center region
        transforms.ToTensor(),          # converts to a float tensor and rescales pixels: 0..255 → 0.0..1.0; converts shape H×W×C → C×H×W.
        transforms.Normalize([0.5]*3,   # same as [0.5, 0.5, 0.5] ; normalizes (0,1) to (-1,1) for all channels
                             [0.5]*3)   # output_c = (input_c - mean[c]) / std[c]
    ])
    return transform


def get_default_dataloader(path=default_dataset_path):
    transform = image_preprocessing_transform() #todo

    dataset = FlatImageFolder(path, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,            # Faster GPU transfer
        persistent_workers=True,    # Keep workers alive between epochs (doubles speed); IMPORTANT True
        #prefetch_factor = 4         # Prefetch 4 batches per worker
    )

    return dataloader


def test_dataloader(dataloader: DataLoader, max_range : int=5):
    for _ in range(max_range):
        images, _ = next(iter(dataloader))

        print(images.shape)
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.axis("off")
        plt.show()

