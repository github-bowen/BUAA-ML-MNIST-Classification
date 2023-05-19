import os
import numpy as np

IMAGE_SIZE = 28 * 28  # MNIST image size


def load_mnist(path, kind='train'):
    """
    'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    Before use, you need to download the above four files to the `path` directory and unzip them
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as label_file:
        labels = np.frombuffer(label_file.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as image_file:
        images = np.frombuffer(image_file.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), IMAGE_SIZE)

    return images, labels
