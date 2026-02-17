#!/usr/bin/env python3
"""
Dataset classes for the evaluation pipeline.
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import numpy as np
import os

class PairedImageDataset(TorchDataset):
    def __init__(self, era5_files, carra2_files, image_size):
        assert len(era5_files) == len(carra2_files)
        self.era5_files = era5_files
        self.carra2_files = carra2_files
        self.image_size = image_size

    def __len__(self):
        return len(self.era5_files)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32)
        arr = arr / 127.5 - 1.0
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        return tensor

    def __getitem__(self, idx):
        return (
            self._load_image(self.era5_files[idx]),
            self._load_image(self.carra2_files[idx]),
            os.path.basename(self.era5_files[idx]),
        )

def collate_with_fnames(batch):
    era5s, carra2s, fnames = zip(*batch)
    era5_batch = torch.stack(era5s, dim=0)
    carra2_batch = torch.stack(carra2s, dim=0)
    return era5_batch, carra2_batch, list(fnames)
