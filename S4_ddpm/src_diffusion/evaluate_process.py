#!/usr/bin/env python3
"""
Main evaluation process.
"""

import os
import glob
import math
import traceback
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import csv

from src_diffusion.evaluate_dataset import PairedImageDataset, collate_with_fnames
from src_diffusion.evaluate_utils import (
    save_image, load_state_dict_from_checkpoint, 
    save_to_netcdf_2d, save_error_trace
)
from src_diffusion.diffusion_dist import create_model_and_diffusion, model_and_diffusion_defaults

def evaluate_process(rank, world_size, args, LOG):
    LOG.info("Starting evaluate_process")

    if world_size is None:
        device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            n_local_gpus = torch.cuda.device_count()
            if local_rank >= n_local_gpus:
                raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {n_local_gpus}. Adjust nproc_per_node or CUDA_VISIBLE_DEVICES.")
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
        LOG.warning(f"Empty dataset shard for rank {rank} (start {start_idx}, end {end_idx}). Syncing and exiting.")
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

    ml_min_img = None
    best_rmse_val = float('inf')

    csv_path = os.path.join(args.output_dir, f"statistics_rank{rank}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["checkpoint","image","rmse","mse"])

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

                    samples_list = []
                    for s in range(args.num_samples):
                        model_kwargs = {args.cond_key: era5} if args.cond_key else {}
                        sample_batch = diffusion.p_sample_loop(
                            model,
                            (B, 3, args.image_size, args.image_size),
                            device=device,
                            model_kwargs=model_kwargs
                        )
                        if isinstance(sample_batch, (list, tuple)):
                            sample_batch = sample_batch[0]
                        samples_list.append(sample_batch)

                    samples_stack = torch.stack(samples_list, dim=0)
                    mse_per_sample = ((samples_stack - era5.unsqueeze(0)) ** 2).mean(dim=[2,3,4])

                    best_idxs = mse_per_sample.argmin(dim=0)
                    arange = torch.arange(B, device=best_idxs.device)
                    best_samples = samples_stack[best_idxs, arange]
                    best_mses = mse_per_sample[best_idxs, arange]

                    rmse_batch = torch.sqrt(best_mses).sum().item()
                    rmse_total += rmse_batch
                    count += B

                    for b in range(B):
                        rmse_val = math.sqrt(best_mses[b].item())
                        mse_val = best_mses[b].item()
                        all_rmses.append(rmse_val)
                        writer.writerow([os.path.basename(ckpt_path), fnames[b], rmse_val, mse_val])

                        if rmse_val < best_rmse_overall:
                            best_rmse_overall = rmse_val
                            best_checkpoint = ckpt_path
                            best_overall_image = best_samples[b].cpu().numpy().transpose(1,2,0)
                            best_overall_fname = fnames[b]
                        
                        if rmse_val < best_rmse_val:
                            best_rmse_val = rmse_val
                            ml_min_img = best_samples[b].cpu().numpy().transpose(1,2,0)

                        out_fname = f"UQ.png"
                        out_path = os.path.join(args.output_dir, out_fname)
                        best_img = best_samples[b].cpu().numpy().transpose(1,2,0)
                        img_array = ((best_img + 1.0) * 127.5).astype(np.uint8)
                        save_image(img_array, out_path)

            rmse_avg = rmse_total / max(count, 1)
            checkpoint_rmses[os.path.basename(ckpt_path)] = all_rmses
            LOG.info(f"Checkpoint {os.path.basename(ckpt_path)}: RMSE={rmse_avg:.4f}")

    LOG.info(f"Best checkpoint: {best_checkpoint} with RMSE={best_rmse_overall:.4f}")

    #
    nc_filename = os.path.join(args.output_dir, "UQ.nc")
    
    if ml_min_img is None:
        raise RuntimeError("ML input are missing; cannot create NetCDF file.")
    
    if len(ml_min_img.shape) == 3:
        ml_min_img_gray = np.dot(ml_min_img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        ml_min_img_gray = ml_min_img
    
    save_to_netcdf_2d(
        ml_min_img_gray, 
        nc_filename, 
        variable_name="UQ", 
        units="jet_normalized", 
        LOG=LOG,
        boundary=5,                    # 
        fill_value=-9999.0,            # 
        normalize_interior=True,       # 
        target_min=0.0,                # 
        target_max=3.0                 # 
    )
    
    # Optional
    if ml_min_img_gray is not None:
        raw_nc_filename = os.path.join(args.output_dir, "UQ_raw.nc")
        save_to_netcdf_2d(
            ml_min_img_gray,
            raw_nc_filename,
            variable_name="UQ_raw",
            units="original_units",
            LOG=LOG,
            boundary=0,                # No boundary processing
            normalize_interior=False   # No normalization
        )
    
    
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
