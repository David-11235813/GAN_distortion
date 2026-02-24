import os
from glob import glob
import numpy as np
from PIL import Image
import my_filesystem as files

IMAGE_FOLDER1 = files.join_path_unsupervised(files.dir_DATASET, "faces_dataset_small_eval_00100")
IMAGE_FOLDER2 = files.join_path_unsupervised(files.dir_DATASET, "faces_dataset_small_eval_01000")
PATTERN = "*.png"
START = 100
COUNT = 100

print("folder: ", IMAGE_FOLDER1)
paths = sorted(glob(os.path.join(IMAGE_FOLDER1, PATTERN)))#[START:START+COUNT]
X = np.vstack([np.asarray(Image.open(p)).ravel().astype(np.float64) for p in paths])
rank = np.linalg.matrix_rank(X)
mean = X.mean()
std = X.std()
print("n_images =", X.shape[0])
print("n_features =", X.shape[1])
print("rank =", rank)
print("mean =", mean)
print("std =", std)


print()
print()

START = 1000
print("folder: ", IMAGE_FOLDER2)
paths = sorted(glob(os.path.join(IMAGE_FOLDER2, PATTERN)))#[START:START+COUNT]
X = np.vstack([np.asarray(Image.open(p)).ravel().astype(np.float64) for p in paths])
rank = np.linalg.matrix_rank(X)
mean = X.mean()
std = X.std()
print("n_images =", X.shape[0])
print("n_features =", X.shape[1])
print("rank =", rank)
print("mean =", mean)
print("std =", std)