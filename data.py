import os
import numpy as np
from imageio import imread
from skimage.transform import resize
from multiprocessing.pool import Pool
from tqdm import tqdm

def _load_image(params):
    path, size = params
    return resize(imread(path), size).astype(np.float32)

def load_images_by_directories(root_path, min_samples, target_size, supported_extensions=[".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
    """Returns a list of images grouped by label. Takes the image directory as its label.
    Works with nested directories."""
    images_by_label = []

    with Pool() as pool:
        for root, _, filenames in tqdm(os.walk(root_path)):
            matching_files = [name for name in filenames if os.path.splitext(name)[1].lower() in supported_extensions]
            if len(matching_files) >= min_samples:
                images = pool.map(_load_image, [(os.path.join(root, path), target_size) for path in matching_files])
                images_by_label.append(np.array(images))

    return images_by_label