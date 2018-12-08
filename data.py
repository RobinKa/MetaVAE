import os
import numpy as np
from imageio import imread
from skimage.transform import resize

def load_images_by_directories(root_path, target_size=(28, 28)):
    """Returns a list of images grouped by label. Takes the image directory as its label.
    Works with nested directories."""
    root_labels = {}

    images_by_label = []

    for root, _, filenames in os.walk(root_path):
        for filename in filenames:
            if not root in root_labels:
                root_labels[root] = len(images_by_label)
                images_by_label.append([])

            image = (imread(os.path.join(root, filename)) / 255.).astype(np.float32)
            image = resize(image, target_size)
            images_by_label[root_labels[root]].append(image)

    images_by_label = [np.array(images) for images in images_by_label]

    return images_by_label