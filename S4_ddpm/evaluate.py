#!/usr/bin/env python3
"""
Evaluation Script for 0-Hour Forecast Diffusion Model

"""
import os
import argparse
import glob
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

from src_diffusion.diffusion_dist import create_model_and_diffusion, model_and_diffusion_defaults

# -------- utils --------
def save_image(img_array, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))
    new_size = (img.width * 4, img.height * 4)
    img = img.resize(new_size, resample=Image.BICUBIC)
    img.save(filename)

def load_state_dict_from_checkpoint(ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    # unwrap common wrappers
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        # if looks like state_dict already
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
        tensor = torch.from_numpy(np.transpose(arr, (2,0,1))).float()
        return tensor

    def __getitem__(self, idx):
        return (
            self._load_image(self.era5_files[idx]),
            self._load_image(self.carra2_files[idx]),
            os.path.basename(self.era5_files[idx]),
        )

# -------- distributed helpers --------
def init_distributed_from_env():
    """
    Initialize via env variables (works with torchrun).
    Returns (rank, world_size) or (None, None) if not running distributed.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        return None, None
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# -------- evaluation logic --------
def evaluate_process(rank, world_size, args):
    # device selection
    if world_size is None:
        device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")

    # load file lists
    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "*.png")))
    carra2_files = sorted(glob.glob(os.path.join(args.carra2_dir, "*.png")))
    print('era5_files =',era5_files)
    print('carra2_files =',carra2_files)
    if len(era5_files) == 0:
        raise RuntimeError("No era5 files found in " + args.era5_dir)
    if len(era5_files) != len(carra2_files):
        raise RuntimeError("ERA5/CARRA2 file count mismatch")

    total_len = len(era5_files)
    print('len(era5_files)=',len(era5_files))
    if world_size is None:
        start_idx, end_idx = 0, total_len
    else:
        per_rank = math.ceil(total_len / world_size)
        start_idx = rank * per_rank
        end_idx = min(start_idx + per_rank, total_len)

    shard_era5 = era5_files[start_idx:end_idx]
    shard_carra2 = carra2_files[start_idx:end_idx]

    dataset = PairedImageDataset(shard_era5, shard_carra2, args.image_size)
    print('dataset',dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model/diffusion (must match training hyperparams)
    defaults = model_and_diffusion_defaults()
    cli_keys = ["image_size","num_channels","num_res_blocks","num_heads","attention_resolutions",
                "dropout","learn_sigma","sigma_small","class_cond","diffusion_steps","noise_schedule"]
    for k in cli_keys:
        v = getattr(args, k, None)
        if v is not None:
            defaults[k] = v
    defaults["image_size"] = args.image_size
    model, diffusion = create_model_and_diffusion(**defaults)
    model.to(device)

    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    print(f"[rank {rank}] Found {len(checkpoint_files)} checkpoints",checkpoint_files)

    best_rmse_overall = float("inf")
    best_checkpoint = None

    for ckpt_path in checkpoint_files:
        try:
            state = load_state_dict_from_checkpoint(ckpt_path)
        except Exception as e:
            print(f"[rank {rank}] Could not read {ckpt_path}: {e}")
            continue

        try:
            model.load_state_dict(state)
        except Exception as e:
            print(f"[rank {rank}] Error loading state into model: {e}")
            print("Ensure model hyperparameters passed here exactly match training.")
            continue
        model.eval()

        rmse_total = 0.0
        count = 0

        with torch.no_grad():
            for era5, carra2, fnames in loader:
                era5 = era5.to(device)
                carra2 = carra2.to(device)
                B = era5.shape[0]

                # accumulate per-sample outputs for the whole batch
                samples_list = []
                for s in range(args.num_samples):
                    # sample a batch at once -> result shape (B, C, H, W)
                    model_kwargs = {args.cond_key: era5} if args.cond_key else {}
                    sample_batch = diffusion.p_sample_loop(model, (B, 3, args.image_size, args.image_size), device=device, model_kwargs=model_kwargs)
                    # ensure tensor
                    if isinstance(sample_batch, (list, tuple)):
                        sample_batch = sample_batch[0]
                    samples_list.append(sample_batch)  # (B, C, H, W)

                # stack -> (S, B, C, H, W)
                samples_stack = torch.stack(samples_list, dim=0)
                # compute mse per sample and per batch item -> (S, B)
                mse_per_sample = ((samples_stack - carra2.unsqueeze(0)) ** 2).mean(dim=[2,3,4])

                # defensive checks
                S = samples_stack.shape[0]
                if S != args.num_samples:
                    print(f"[rank {rank}] WARNING: generated S={S} samples but requested {args.num_samples}.")

                # best index per batch item: shape (B,)
                best_idxs = mse_per_sample.argmin(dim=0)   # dtype long

                # gather best samples for each batch item using advanced indexing
                arange = torch.arange(B, device=best_idxs.device)
                best_samples = samples_stack[best_idxs, arange]  # (B, C, H, W)
                best_mses = mse_per_sample[best_idxs, arange]    # (B,)

                # update metrics
                rmse_batch = torch.sqrt(best_mses).sum().item()
                rmse_total += rmse_batch
                count += B

                # save each best sample individually
                for b in range(B):
                    basename = fnames[b] if isinstance(fnames, (list, tuple)) else fnames
                    out_fname = f"{os.path.splitext(basename)[0]}__ckpt_{os.path.basename(ckpt_path)}.png"
                    out_path = os.path.join(args.output_dir, out_fname)
                    best_img = best_samples[b].cpu().numpy().transpose(1,2,0)
                    img_array = ((best_img + 1.0) * 127.5).astype(np.uint8)
                    save_image(img_array, out_path)

        rmse_avg = rmse_total / max(count, 1)
        print(f"[rank {rank}] Checkpoint {os.path.basename(ckpt_path)}: RMSE={rmse_avg:.4f}")

        if rmse_avg < best_rmse_overall:
            best_rmse_overall = rmse_avg
            best_checkpoint = ckpt_path

    print(f"[rank {rank}] Best checkpoint: {best_checkpoint} with RMSE={best_rmse_overall:.4f}")

    if world_size is not None:
        dist.barrier()

# -------- main --------
def main():
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
    # optional model overrides
    parser.add_argument("--num_channels", type=int, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--attention_resolutions", type=str, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # init distributed if launched with torchrun
    rank, world_size = init_distributed_from_env()

    if rank is None:
        print("Running single-process mode.")
        evaluate_process(0, None, args)
    else:
        print(f"Distributed init OK - rank {rank} / world_size {world_size}")
        evaluate_process(rank, world_size, args)
        cleanup_distributed()

if __name__ == "__main__":
    main()
