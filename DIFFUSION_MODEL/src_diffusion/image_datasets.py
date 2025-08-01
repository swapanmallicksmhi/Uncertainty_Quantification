"""
Image Data Loader for Deep Learning Training Pipelines.

This module provides functionality for:
- Recursively loading image files from a directory.
- Applying class-based sharding (useful for distributed training via MPI).
- Preprocessing images (resizing, cropping, normalization).
- Creating PyTorch-style Dataset and DataLoader objects.

Dependencies:
- PIL (Pillow)
- numpy
- torch
- mpi4py
- blobfile (for GCS/local filesystem compatibility)

Author: Swapan Mallick, SMHI
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from mpi4py import MPI
import blobfile as bf


def load_data(
    *,
    data_dir: str,
    batch_size: int,
    image_size: int,
    class_cond: bool = False,
    deterministic: bool = False,
):
    """
    Load data from a directory into a DataLoader that yields (image, metadata) pairs.

    Args:
        data_dir (str): Path to directory containing image files.
        batch_size (int): Number of samples per batch.
        image_size (int): Final resolution of images (square).
        class_cond (bool): If True, extract class labels from filenames.
        deterministic (bool): If True, disable shuffling for reproducibility.

    Yields:
        (image, metadata) tuples. Images are normalized float32 NCHW arrays.
        Metadata is a dict containing optional class labels ('y').
    """
    if not data_dir:
        raise ValueError("You must specify a valid data directory.")

    all_files = _list_image_files_recursively(data_dir)
    classes = None

    if class_cond:
        # Class is determined by the prefix of the filename before the first underscore.
        class_names = [bf.basename(path).split("carra2")[0] for path in all_files] # SWAPAN
        sorted_classes = {name: idx for idx, name in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[name] for name in class_names]

    dataset = ImageDataset(
        resolution=image_size,
        image_paths=all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir: str) -> list:
    """
    Recursively collect image files from a directory.

    Args:
        data_dir (str): Root directory.

    Returns:
        list: Paths to image files (jpg, jpeg, png, gif).
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1].lower()

        if "." in entry and ext in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))

    return results


class ImageDataset(Dataset):
    """
    PyTorch-style Dataset for loading and preprocessing images.

    Args:
        resolution (int): Final image resolution.
        image_paths (list): List of image file paths.
        classes (list, optional): List of integer class labels.
        shard (int): Rank of the current MPI process.
        num_shards (int): Total number of MPI processes.
    """

    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard::num_shards]
        self.local_classes = (
            None if classes is None else classes[shard::num_shards]
        )

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # Manual downsampling using BOX filter if the image is too large
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                (pil_image.size[0] // 2, pil_image.size[1] // 2), resample=Image.BOX
            )

        # Resize to preserve aspect ratio
        scale = self.resolution / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=Image.BICUBIC)

        # Convert image to numpy array and crop to square
        arr = np.array(pil_image.convert("RGB"), dtype=np.float32)
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]

        # Normalize to [-1, 1]
        arr = arr / 127.5 - 1.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # Convert from HWC to CHW
        return np.transpose(arr, (2, 0, 1)), out_dict
