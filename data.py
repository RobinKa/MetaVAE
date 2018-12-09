import os
import numpy as np
from imageio import imread
from skimage.transform import resize
from multiprocessing.pool import Pool
from tqdm import tqdm

def _load_image(params):
    path, size = params
    return resize(imread(path), size).astype(np.float32)

def load_images_by_directories(root_path, min_samples, target_size):
    """Returns a list of images grouped by label. Takes the image directory as its label.
    Works with nested directories."""
    images_by_label = []

    with Pool() as pool:
        for root, _, filenames in tqdm(os.walk(root_path)):
            if len(filenames) >= min_samples:
                images = pool.map(_load_image, [(os.path.join(root, path), target_size) for path in filenames])
                images_by_label.append(np.array(images))

    return images_by_label