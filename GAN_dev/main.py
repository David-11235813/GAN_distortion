#pip install -r req_dev.txt

import os
from typing import List
from PIL import Image
import torch


def log(text) :
    print(text)

def list_jpg_files(imagesPath: os.path) -> List[str]:
    return sorted(f for f in os.listdir(imagesPath)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(imagesPath, f)))


def access_dataset():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'dataset'))
    img_folder_path = os.path.join(dataset_path, 'PhotosColorPicker')
    log(img_folder_path)

    images = list_jpg_files(img_folder_path)
    log(images)

    f = 1
    for image in images:
        jpg = Image.open(os.path.join(img_folder_path, image))
        jpg.show()

        if f >= 3: break
        f+=1



if __name__ == '__main__':
    access_dataset()

'''
    checkpoint_path = f"best_checkpoint_X_colors.pth"

    # Checkpoint & Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        print(f"Epoch {epoch}: New best val_loss {best_val_loss:.4f} - saving checkpoint.")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "metrics": metrics,
        }, checkpoint_path)
    else:
        no_improve += 1


    # Optionally load best model for final evaluation / inference:
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with val_loss {ckpt['best_val_loss']:.4f}")
'''