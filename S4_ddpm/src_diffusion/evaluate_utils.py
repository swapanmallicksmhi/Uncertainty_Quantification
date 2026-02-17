#!/usr/bin/env python3
"""
Utility functions for the evaluation pipeline.
"""
import os
import sys
import logging
import traceback
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator


def setup_rank_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"rank_{rank}.log")

    logger_name = f"evaluate.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path)

    fmt_str = f"%(asctime)s [rank {rank}] %(levelname)s: %(message)s"
    fmt = logging.Formatter(fmt_str)

    stream_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def save_error_trace(output_dir, rank, tb):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"error_rank{rank}.log")
    with open(path, "w") as f:
        f.write(tb)
    return path

def save_image(img_array, filename, data=None, upscale_factor=4, dpi=100):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    new_width = img.width * upscale_factor
    new_height = img.height * upscale_factor
    new_size = (new_width, new_height)

    img_upscaled = img.resize(new_size, resample=Image.BICUBIC)

    fig, ax = plt.subplots(figsize=(new_width/100, new_height/100), dpi=100)

    if data is not None and 'x' in data and 'y' in data:
        x0, x1 = data['x'].min(), data['x'].max()
        y0, y1 = data['y'].min(), data['y'].max()

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        domain_width = x1 - x0
        domain_height = y1 - y0
        if domain_height > 0:
            aspect_ratio = domain_width / domain_height
            ax.set_aspect(aspect_ratio)

    ax.imshow(np.array(img_upscaled), extent=[x0, x1, y0, y1] if data else None)
    ax.axis('off')

    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        dpi=dpi,
        transparent=False
    )
    plt.close(fig)

def save_image_simple1(img_array, filename, upscale_factor=4):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    new_width = img.width * upscale_factor
    new_height = img.height * upscale_factor
    new_size = (new_width, new_height)

    img_upscaled = img.resize(new_size, resample=Image.BICUBIC)
    img_upscaled.save(filename, quality=95)

def load_state_dict_from_checkpoint(ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            return state
    raise RuntimeError(f"Checkpoint {ckpt_path} doesn't contain a usable state_dict.")

def normalize_range(data, target_min=0.0, target_max=3.0):
    """
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max > data_min:
        normalized = target_min + (data - data_min) * (target_max - target_min) / (data_max - data_min)
    else:
        normalized = np.full_like(data, (target_min + target_max) / 2.0)
    
    return normalized, data_min, data_max

def process_with_boundary_normalization(data, boundary=5, fill_value=-9999.0, 
                                        normalize=True, target_min=0.0, target_max=3.0):
    """
    """
    ny, nx = data.shape
    
    # Create output array with fill_value
    output = np.full((ny, nx), fill_value, dtype=np.float32)
    
    # Calculate interior region
    y_start, y_end = boundary, ny - boundary
    x_start, x_end = boundary, nx - boundary
    
    # Check if boundary size is valid
    if y_start >= y_end or x_start >= x_end:
        raise ValueError(f"Boundary {boundary} too large for data dimensions {ny}x{nx}")
    
    # Extract interior
    interior = data[y_start:y_end, x_start:x_end]
    
    # Normalize interior if requested
    if normalize and interior.size > 0:
        interior_norm, orig_min, orig_max = normalize_range(interior, target_min, target_max)
    else:
        interior_norm = interior.copy()
        orig_min = np.min(interior) if interior.size > 0 else 0.0
        orig_max = np.max(interior) if interior.size > 0 else 0.0
    
    # Put processed interior into output array
    output[y_start:y_end, x_start:x_end] = interior_norm
    
    # Calculate statistics
    fill_count = np.sum(output == fill_value)
    valid_count = np.sum(output != fill_value)
    
    stats = {
        'original_shape': (ny, nx),
        'original_min': float(np.min(data)),
        'original_max': float(np.max(data)),
        'original_mean': float(np.mean(data)),
        'interior_shape': (y_end - y_start, x_end - x_start),
        'interior_min': float(orig_min),
        'interior_max': float(orig_max),
        'normalized_min': float(np.min(interior_norm)) if interior_norm.size > 0 else 0.0,
        'normalized_max': float(np.max(interior_norm)) if interior_norm.size > 0 else 0.0,
        'fill_count': int(fill_count),
        'valid_count': int(valid_count),
        'boundary_size': boundary,
        'fill_value': fill_value,
        'normalized': normalize,
        'target_range': (target_min, target_max)
    }
    
    return output, stats
#------------------------------NetCDF file out------------------
def save_to_netcdf_2d(
    data,
    filename,
    variable_name="UQ",
    units="Kelvin",
    LOG=None,
    boundary=0,
    fill_value=None,
    normalize_interior=False,
    target_min=0.0,
    target_max=3.0,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if data is None:
        raise RuntimeError(f"No data provided for NetCDF file: {filename}")

    # =========================================================
    # Optional boundary processing
    # =========================================================
    stats = None
    if boundary > 0:
        if fill_value is None:
            fill_value = -9999.0

        processed_data, stats = process_with_boundary_normalization(
            data,
            boundary,
            fill_value,
            normalize_interior,
            target_min,
            target_max,
        )
    else:
        processed_data = data.copy()
        if fill_value is None:
            fill_value = -9999.0

    # =========================================================
    # Regrid to high resolution (2880 x 2880)
    # =========================================================
    ny, nx = processed_data.shape
    TARGET_Y, TARGET_X = 2880, 2880

    y_src = np.linspace(0, ny - 1, ny)
    x_src = np.linspace(0, nx - 1, nx)
    y_tgt = np.linspace(0, ny - 1, TARGET_Y)
    x_tgt = np.linspace(0, nx - 1, TARGET_X)

    masked = np.ma.masked_equal(processed_data, fill_value)
    src_float = masked.filled(np.nan)

    interp = RegularGridInterpolator(
        (y_src, x_src),
        src_float,
        bounds_error=False,
        fill_value=np.nan,
    )

    Y, X = np.meshgrid(y_tgt, x_tgt, indexing="ij")
    points = np.stack([Y.ravel(), X.ravel()], axis=-1)

    interp_data = interp(points).reshape(TARGET_Y, TARGET_X)
    interp_data = np.where(np.isnan(interp_data), fill_value, interp_data)

    # =========================================================
    # Write NetCDF
    # =========================================================
    with nc.Dataset(filename, "w", format="NETCDF4") as ncd:
        ncd.createDimension("y", TARGET_Y)
        ncd.createDimension("x", TARGET_X)

        var = ncd.createVariable(
            variable_name,
            "f4",
            ("y", "x"),
            fill_value=fill_value,
            zlib=True,
            complevel=4,
        )

        var[:, :] = np.flipud(interp_data)
        var.units = "Kelvin"
        var.resolution = "2880x2880"

        # =====================================================
        # CLEAN VARIABLE ATTRIBUTES (REMOVE UNNECESSARY)
        # =====================================================
        REMOVE_VAR_ATTRS = {
            "normalized_interior",
            "original_min",
            "original_max",
            "interior_shape",
            "valid_cell_count",
            "fill_cell_count",
            "normalization_range",
            "jet_colormap_mapping",
            "normalized_min",
            "normalized_max",
            "original_resolution",
            "interpolation",
            "boundary_size",
        }

        for attr in list(var.ncattrs()):
            if attr in REMOVE_VAR_ATTRS:
                delattr(var, attr)

        # =====================================================
        # CLEAN GLOBAL ATTRIBUTES
        # =====================================================
        ncd.description = "Generated uncertainty quantification data"
        ncd.source = "Diffusion model evaluation script"
        ncd.created = str(np.datetime64("now"))
        ncd.created_by = "DDPM-ML model"

    if LOG:
        LOG.info(f"High-resolution NetCDF saved: {filename}")
        #---------------------------------------------------------------------------------------

def init_distributed_from_env():
    rank = None
    world_size = None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        return None, None

    try:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    except Exception as e:
        raise RuntimeError(f"Failed to init process group: {e}")
    
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    except Exception:
        pass
    
    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

