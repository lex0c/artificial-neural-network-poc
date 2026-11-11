import gzip
import hashlib
import os
import urllib.request

import numpy as np


# MNIST: 70,000 images of handwritten digits (0 to 9), 28 x 28 pixels in shades of gray, split into 60,000 images for
# training and 10,000 for testing. load_mnist() downloads the original files once (about 11 MB) into datasets/mnist/.

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"  # a copy of the original files

# file name and SHA-256 of each file, to detect a corrupted download
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
}


def download(filename, sha256, directory):
    path = os.path.join(directory, filename)

    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        print(f"Downloading {URL + filename}")
        urllib.request.urlretrieve(URL + filename, path + ".part")
        os.replace(path + ".part", path)

    with open(path, "rb") as file:
        if hashlib.sha256(file.read()).hexdigest() != sha256:
            raise ValueError(f"{path} is corrupted (unexpected SHA-256): delete it to download it again")

    return path


# The IDX format of the MNIST files: a 4-byte "magic number" whose last byte is the number of dimensions, the size of
# each dimension (4-byte big-endian integers), and then the values, one byte each (a pixel from 0 to 255, or a label).
def read_idx(path):
    with gzip.open(path, "rb") as file:
        raw = file.read()

    num_dimensions = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(num_dimensions)]

    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * num_dimensions).reshape(shape)


# Returns (train_images, train_labels), (test_images, test_labels). Each image is flattened to 784 values (the 28 x 28
# pixels, row by row) scaled from 0-255 to 0-1, and each label is the digit.
def load_mnist(directory=os.path.join("datasets", "mnist")):
    data = {name: read_idx(download(filename, sha256, directory)) for name, (filename, sha256) in FILES.items()}

    def flatten(images):
        return images.reshape(len(images), -1) / 255.0

    return (
        (flatten(data["train_images"]), data["train_labels"].astype(int)),
        (flatten(data["test_images"]), data["test_labels"].astype(int)),
    )
