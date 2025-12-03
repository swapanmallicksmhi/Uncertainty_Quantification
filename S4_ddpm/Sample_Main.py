####!/usr/bin/env python3
"""

Main program for Sampling.
Author: Swapan Mallick
Date : 9 March 2025
Sampler for DDPM 0-h forecast models.

"""
import argparse
import os
import sys
import traceback
import torch
import numpy as np
from PIL import Image

from src_diffusion.diffusion_dist import create_model_and_diffusion, model_and_diffusion_defaults

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def save_images(batch_tensor, out_dir, prefix="sample", save_size=None):
    """
    Save images from batch_tensor to out_dir.
    batch_tensor: torch.Tensor (B, C, H, W) in [-1, 1]
    save_size: int or None, if set images are resized to (save_size, save_size)
    """
    batch_tensor = batch_tensor.clamp(-1, 1)
    imgs = ((batch_tensor.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    for i, arr in enumerate(imgs):
        im = Image.fromarray(arr)
        if save_size is not None and save_size != arr.shape[0]:
            im = im.resize((save_size, save_size), resample=Image.BICUBIC)
        im.save(os.path.join(out_dir, f"{prefix}_{i}.png"))


def load_checkpoint(model, model_path):
    """
    Load checkpoint into model. Handles common container forms and
    returns True on success. On failure prints helpful diagnostics.
    """
    # Try torch.load with weights_only if available (suppresses future warning)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        # older PyTorch doesn't have weights_only
        state = torch.load(model_path, map_location="cpu")
    except Exception:
        print("Failed to load checkpoint. Traceback:")
        traceback.print_exc()
        return False

    # If the checkpoint is a dict containing 'state_dict' or 'model' etc., try to find it
    if isinstance(state, dict):
        candidate_keys = ["state_dict", "model_state_dict", "model", "state"]
        found = None
        for k in candidate_keys:
            if k in state:
                found = k
                break
        if found:
            print(f"[INFO] Found top-level key '{found}' in checkpoint; using that sub-dict.")
            state = state[found]

    # state should now be a mapping name -> tensor (a state_dict)
    if not isinstance(state, dict):
        print("Checkpoint does not contain a state_dict (found type {}).".format(type(state)))
        return False

    # Try to load
    try:
        model.load_state_dict(state)
        return True
    except RuntimeError as e:
        # Provide detailed mismatch diagnostics
        print("RuntimeError while loading state_dict (likely architecture mismatch).")
        print(e)
        return False
    except Exception:
        print("Unexpected exception when loading checkpoint:")
        traceback.print_exc()
        return False


def build_model_from_args(args):
    defaults = model_and_diffusion_defaults()

    override_keys = [
        "image_size", "num_channels", "num_res_blocks", "num_heads",
        "attention_resolutions", "dropout", "learn_sigma", "sigma_small",
        "class_cond", "diffusion_steps", "noise_schedule", "timestep_respacing",
        "use_kl", "predict_xstart", "rescale_timesteps", "rescale_learned_sigmas",
        "use_checkpoint", "use_scale_shift_norm"
    ]
    for key in override_keys:
        val = getattr(args, key, None)
        if val is not None:
            # convert boolean-like strings
            if isinstance(defaults.get(key, None), bool):
                defaults[key] = str2bool(val) if isinstance(val, str) else val
            else:
                defaults[key] = val

    # attention_resolutions might be passed as string "16,8"
    if isinstance(defaults.get("attention_resolutions", None), str):
        defaults["attention_resolutions"] = defaults["attention_resolutions"]

    print("Model/diffusion kwargs:", defaults)
    model, diffusion = create_model_and_diffusion(**defaults)
    return model, diffusion


def sample_main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.SAMPLE_OUT, exist_ok=True)

    # Build model with provided overrides
    model, diffusion = build_model_from_args(args)

    # Load checkpoint into model
    CK_OK = load_checkpoint(model, args.model_path)
    if not CK_OK:
        print("Could not load the checkpoint into the model. Aborting.")
        sys.exit(2)

    # Move model to device
    model.to(device)
    model.eval()

    # sampling loop: generate num_samples images in batches
    n_done = 0
    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random

    while n_done < args.num_samples:
        this_batch = min(args.batch_size, args.num_samples - n_done)
        shape = (this_batch, 3, args.image_size, args.image_size)
        print(f"[INFO] Sampling batch of {this_batch}, target total {args.num_samples}")

        with torch.no_grad():
            # sampling uses diffusion object's sampling method; this assumes diffusion has p_sample_loop
            samples = diffusion.p_sample_loop(model, shape, device=device)

        save_images(samples, args.SAMPLE_OUT, prefix=f"sample_{n_done}", save_size=args.save_size)
        n_done += this_batch

    print("Sampling done. Saved samples to:", args.SAMPLE_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--SAMPLE_OUT", required=True)
    parser.add_argument("--image_size", type=int, default=None, help="Model image size")
    parser.add_argument("--save_size", type=int, default=None, help="Save images at this resolution (upscaled)")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Model / diffusion overrides (SM)
    parser.add_argument("--num_channels", type=int, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--attention_resolutions", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learn_sigma", type=str, default=None)
    parser.add_argument("--sigma_small", type=str, default=None)
    parser.add_argument("--class_cond", type=str, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)
    parser.add_argument("--timestep_respacing", type=str, default=None)
    parser.add_argument("--use_kl", type=str, default=None)
    parser.add_argument("--predict_xstart", type=str, default=None)
    parser.add_argument("--rescale_timesteps", type=str, default=None)
    parser.add_argument("--rescale_learned_sigmas", type=str, default=None)
    parser.add_argument("--use_checkpoint", type=str, default=None)
    parser.add_argument("--use_scale_shift_norm", type=str, default=None)

    args = parser.parse_args()

    # If image_size not passed, fall back to default from model defaults
    if args.image_size is None:
        args.image_size = model_and_diffusion_defaults().get("image_size", 256)

    sample_main(args)
