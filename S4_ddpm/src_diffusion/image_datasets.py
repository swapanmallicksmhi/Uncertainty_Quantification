"""
Image Data Loader for Deep Learning Training Pipelines (ERA5 -> CARRA2 0h forecast)

Author: Swapan Mallick
"""

import os
from typing import List, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import blobfile as bf

# ---------- Single image dataset ----------
class SingleImageDataset(Dataset):
    def __init__(self, resolution, image_paths: List[str]):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        tensor = self._process_image(self.image_paths[idx])
        return tensor, {}

    def _process_image(self, path: str) -> torch.FloatTensor:
        with bf.BlobFile(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()

# ---------- Paired dataset ----------
class PairedDataset0h(Dataset):
    """
    Pairs ERA5 images with CARRA2 images at the same timestamp (0-hour forecast).
    """
    def __init__(self, era5_paths: List[str], carra2_paths: List[str], resolution: int):
        super().__init__()
        self.resolution = resolution
        self.pairs = self._make_pairs(era5_paths, carra2_paths)

    def _make_pairs(self, era5_paths, carra2_paths):
        pairs = []
        # Map timestamp prefix to path
        carra_dict = {}
        for p in carra2_paths:
            base = os.path.basename(p).split("_")[-2]  # YYYYMMDDHH
            carra_dict[base] = p

        for e in era5_paths:
            parts = os.path.basename(e).split("_")
            ts = parts[-2]  # YYYYMMDDHH
            # match with same timestamp (0h difference)
            if ts in carra_dict:
                pairs.append((e, carra_dict[ts]))
        if len(pairs) == 0:
            raise RuntimeError("No ERA5->CARRA2 0h pairs found.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        era5_path, carra2_path = self.pairs[idx]
        return self._process_image(era5_path), self._process_image(carra2_path)

    def _process_image(self, path: str) -> torch.FloatTensor:
        with bf.BlobFile(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()

def load_data(data_dir: str, batch_size: int, image_size: int):
    all_files = [bf.join(data_dir, f) for f in bf.listdir(data_dir) if f.endswith(".png")]
    era5_files = [f for f in all_files if "era5" in f]
    carra2_files = [f for f in all_files if "carra2" in f]
    dataset = PairedDataset0h(era5_files, carra2_files, resolution=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader
