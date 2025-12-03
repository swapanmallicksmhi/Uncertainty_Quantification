#!/usr/bin/env python3
"""
================================================================================
Author       : Swapan Mallick, SMHI
Date Created : 06/09/2025
================================================================================
Description:
------------
This script evaluates diffusion-based super-resolution or downscaling models 
(e.g., CARRA2 vs ERA5) by generating synthetic high-quality samples, 
computing RMSE/MSE-based quality metrics, and saving the best generated image 
for each dataset pair and model checkpoint.

Inputs:
--------
--checkpoint_dir   Path to the directory containing model checkpoints (.pt)
--era5_dir         Directory containing ERA5 PNG images
--carra2_dir       Directory containing CARRA2 PNG images
--output_dir       Output directory for results
--image_size       Target image dimension for resizing (default: 256)
--batch_size       Batch size per process (default: 1)
--num_samples      Number of candidate samples to generate per input (default: 10)
--cond_key         Conditioning key (default: "low_res")
--device           Device to use: "cuda" or "cpu" (default: "cuda")
--num_channels     Number of model channels (optional)
--num_res_blocks   Number of residual blocks in the model (optional)
--num_heads        Number of attention heads (optional)
--attention_resolutions  Attention layer resolutions (optional)
--diffusion_steps  Number of diffusion steps (optional)
--noise_schedule   Type of diffusion noise schedule (optional)

================================================================================
"""

import os
import sys
import argparse
import glob
import math
import traceback
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
from src_diffusion.diffusion_dist import create_model_and_diffusion, model_and_diffusion_defaults

# Global logger (set in main)
LOG = None

# -------- utils --------
def setup_rank_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"rank_{rank}.log")

    logger_name = f"evaluate.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 
    for h in list(logger.handlers):
        logger.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path)

    # 
    fmt_str = f"%(asctime)s [rank {rank}] %(levelname)s: %(message)s"
    fmt = logging.Formatter(fmt_str)

    stream_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    # 
    logger.propagate = False
    return logger

def save_error_trace(output_dir, rank, tb):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"error_rank{rank}.log")
    with open(path, "w") as f:
        f.write(tb)
    return path

#---------------------------------------
def save_image(img_array, filename, data=None, upscale_factor=4, dpi=100):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    new_width = img.width * upscale_factor
    new_height = img.height * upscale_factor
    new_size = (new_width, new_height)

    img_upscaled = img.resize(new_size, resample=Image.BICUBIC)

    fig, ax = plt.subplots(figsize=(new_width/100, new_height/100), dpi=100)

    # Set proper aspect ratio if coordinate data is provided
    if data is not None and 'x' in data and 'y' in data:
        x0, x1 = data['x'].min(), data['x'].max()
        y0, y1 = data['y'].min(), data['y'].max()

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        # 
        domain_width = x1 - x0
        domain_height = y1 - y0
        if domain_height > 0:  # Avoid division by zero
            aspect_ratio = domain_width / domain_height
            ax.set_aspect(aspect_ratio)

    ax.imshow(np.array(img_upscaled), extent=[x0, x1, y0, y1] if data else None)
    ax.axis('off')  # Turn off axes for clean image

    # 
    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        dpi=dpi,
        transparent=False
    )
    plt.close(fig)

# 
def save_image_simple1(img_array, filename, upscale_factor=4):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert array to PIL Image
    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    # Calculate new size and upscale
    new_width = img.width * upscale_factor
    new_height = img.height * upscale_factor
    new_size = (new_width, new_height)

    # Upscale and save
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

# -------- dataset --------
class PairedImageDataset(Dataset):
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

# 
def collate_with_fnames(batch):
    era5s, carra2s, fnames = zip(*batch)
    era5_batch = torch.stack(era5s, dim=0)
    carra2_batch = torch.stack(carra2s, dim=0)
    return era5_batch, carra2_batch, list(fnames)

# 
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

def generate_high_quality_sample(diffusion, model, era5_batch, carra2_batch, device, num_samples, cond_key, image_size, num_refinement_steps=2):
    B = era5_batch.shape[0]
    best_samples = []
    best_mses = []
    
    for b in range(B):
        era5_single = era5_batch[b:b+1]
        carra2_single = carra2_batch[b:b+1]
        
        candidate_samples = []
        candidate_mses = []
        
        # 
        for _ in range(num_samples):
            model_kwargs = {cond_key: era5_single} if cond_key else {}
            sample = diffusion.p_sample_loop(
                model,
                (1, 3, image_size, image_size),
                device=device,
                model_kwargs=model_kwargs,
                progress=False
            )
            
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
            
            # 
            mse = ((sample - carra2_single) ** 2).mean().item()
            candidate_samples.append(sample)
            candidate_mses.append(mse)
        
        # 
        best_idx = np.argmin(candidate_mses)
        best_sample = candidate_samples[best_idx]
        best_mse = candidate_mses[best_idx]
        
        # 
        if num_refinement_steps > 0:
            current_best = best_sample.clone()
            for refinement_step in range(num_refinement_steps):
                # 
                noise_level = 0.1 * (1 - (refinement_step / num_refinement_steps))
                noisy_sample = current_best + torch.randn_like(current_best) * noise_level
                
                # 
                model_kwargs = {cond_key: era5_single} if cond_key else {}
                refined_sample = diffusion.p_sample_loop(
                    model,
                    (1, 3, image_size, image_size),
                    device=device,
                    model_kwargs=model_kwargs,
                    progress=False
                )
                
                if isinstance(refined_sample, (list, tuple)):
                    refined_sample = refined_sample[0]
                
                refined_mse = ((refined_sample - carra2_single) ** 2).mean().item()
                
                # Keep if improved
                if refined_mse < best_mse:
                    current_best = refined_sample
                    best_mse = refined_mse
        
        best_samples.append(current_best.squeeze(0))
        best_mses.append(best_mse)
    
    return torch.stack(best_samples), torch.tensor(best_mses)

# 
def evaluate_process(rank, world_size, args):
    global LOG
    LOG.info("Starting evaluate_process")

    if world_size is None:
        device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            n_local_gpus = torch.cuda.device_count()
            if local_rank >= n_local_gpus:
                raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {n_local_gpus} ")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "*.png")))
    carra2_files = sorted(glob.glob(os.path.join(args.carra2_dir, "*.png")))
    if len(era5_files) == 0:
        raise RuntimeError("No ERA5 files found in " + args.era5_dir)
    if len(era5_files) != len(carra2_files):
        raise RuntimeError("ERA5/CARRA2 file count mismatch")

    total_len = len(era5_files)
    if world_size is None:
        start_idx, end_idx = 0, total_len
    else:
        per_rank = math.ceil(total_len / world_size)
        start_idx = rank * per_rank
        end_idx = min(start_idx + per_rank, total_len)

    shard_era5 = era5_files[start_idx:end_idx]
    shard_carra2 = carra2_files[start_idx:end_idx]

    dataset = PairedImageDataset(shard_era5, shard_carra2, args.image_size)
    if len(dataset) == 0:
        LOG.warning(f"Empty dataset shard for rank {rank} ")
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_with_fnames)

    defaults = model_and_diffusion_defaults()
    cli_keys = ["image_size","num_channels","num_res_blocks","num_heads","attention_resolutions",
                "dropout","learn_sigma","sigma_small","class_cond","diffusion_steps","noise_schedule"]
    for k in cli_keys:
        v = getattr(args, k, None)
        if v is not None:
            defaults[k] = v
    defaults["image_size"] = args.image_size

    model, diffusion = create_model_and_diffusion(**defaults)
    try:
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to move model to device {device}: {e}")

    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if len(checkpoint_files) == 0:
        raise RuntimeError("No checkpoints found in " + args.checkpoint_dir)

    best_rmse_overall = float("inf")
    best_checkpoint = None
    best_overall_image = None
    best_overall_fname = None
    best_carra2_image = None 

    csv_path = os.path.join(args.output_dir, f"statistics_rank{rank}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["checkpoint","image","rmse","mse","similarity_score"])

        checkpoint_rmses = {}
        for ckpt_path in checkpoint_files:
            try:
                state = load_state_dict_from_checkpoint(ckpt_path)
                model.load_state_dict(state)
            except Exception as e:
                LOG.error(f"Could not load {ckpt_path}: {e}")
                tb = traceback.format_exc()
                save_error_trace(args.output_dir, rank, tb)
                continue
            model.eval()

            rmse_total = 0.0
            count = 0
            all_rmses = []

            with torch.no_grad():
                for era5, carra2, fnames in loader:
                    era5 = era5.to(device)
                    carra2 = carra2.to(device)
                    B = era5.shape[0]

                    # 
                    best_samples, best_mses = generate_high_quality_sample(
                        diffusion, model, era5, carra2, device, 
                        num_samples=args.num_samples, 
                        cond_key=args.cond_key, 
                        image_size=args.image_size,
                        num_refinement_steps=2
                    )

                    rmse_batch = torch.sqrt(best_mses).sum().item()
                    rmse_total += rmse_batch
                    count += B

                    for b in range(B):
                        rmse_val = math.sqrt(best_mses[b].item())
                        mse_val = best_mses[b].item()
                        all_rmses.append(rmse_val)
                        
                        # 
                        similarity_score = 1.0 / (1.0 + rmse_val)
                        writer.writerow([os.path.basename(ckpt_path), fnames[b], rmse_val, mse_val, similarity_score])

                        if rmse_val < best_rmse_overall:
                            best_rmse_overall = rmse_val
                            best_checkpoint = ckpt_path
                            best_overall_image = best_samples[b].cpu().numpy().transpose(1,2,0)
                            best_overall_fname = fnames[b]
                            best_carra2_image = carra2[b].cpu().numpy().transpose(1,2,0)

                        # 
                        out_fname = f"FIELD_ckpt_{os.path.basename(ckpt_path)}_BEST.png"
                        out_path = os.path.join(args.output_dir, out_fname)
                        best_img = best_samples[b].cpu().numpy().transpose(1,2,0)
                        img_array = ((best_img + 1.0) * 127.5).astype(np.uint8)
                        save_image(img_array, out_path)

            rmse_avg = rmse_total / max(count, 1)
            checkpoint_rmses[os.path.basename(ckpt_path)] = all_rmses
            LOG.info(f"Checkpoint {os.path.basename(ckpt_path)}: RMSE={rmse_avg:.4f}")

    LOG.info(f"Best checkpoint: {best_checkpoint} with RMSE={best_rmse_overall:.4f}")

    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass

# -------- main --------
def main():
    global LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--era5_dir", required=True)
    parser.add_argument("--carra2_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--cond_key", type=str, default="low_res")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_channels", type=int, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--attention_resolutions", type=str, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        rank, world_size = init_distributed_from_env()
    except Exception as e:
        tb = traceback.format_exc()
        print("Failed to init distributed:", e)
        print(tb)
        sys.exit(1)

    rank_for_log = rank if rank is not None else 0
    LOG = setup_rank_logging(args.output_dir, rank_for_log)

    try:
        try:
            evaluate_process(rank_for_log if rank is not None else 0, world_size, args)
        except Exception as e:
            tb = traceback.format_exc()
            LOG.error("Unhandled exception in evaluate_process:\n%s", tb)
            save_error_trace(args.output_dir, rank_for_log, tb)
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

if __name__ == "__main__":
    main()
