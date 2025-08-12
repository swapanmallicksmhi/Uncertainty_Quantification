#-
#Author: Swapan Mallick
#Date : 9 March 2025
#-

import argparse
import os
import sys
import numpy as np
import torch as th
import torch.distributed as dist

# Extend system path to import from parent and current directories
sys.path.extend(["..", "."])

from src_diffusion import diffusion_dist, logger
from src_diffusion.diffusion_script import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    # Setup distributed training environment
    diffusion_dist.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion process...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load the trained model
    model.load_state_dict(
        diffusion_dist.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(diffusion_dist.dev())
    model.eval()

    logger.log("Sampling started...")
    all_images = []
    all_labels = []

    # Generate samples until the desired number is reached
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}

        # Add class conditioning if enabled
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=diffusion_dist.dev()
            )
            model_kwargs["y"] = classes

        # Choose sampling method
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Generate samples
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # Post-process the sample: scale to [0, 255] and convert to HWC format
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        # Gather samples from all distributed processes
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

        # Gather class labels if class conditioning is used
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_labels])

        logger.log(f"Created {len(all_images) * args.batch_size} samples")

    # Combine samples and trim to the exact number requested
    arr = np.concatenate(all_images, axis=0)[:args.num_samples]

    # Combine and trim labels if present
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)[:args.num_samples]

    # Save only from the main process (rank 0)
    if dist.get_rank() == 0:
        out_dir = args.SAMPLE_OUT
        shape_str = "x".join(map(str, arr.shape))
        out_path = os.path.join(out_dir, f"samples_{shape_str}.npz")
        print('out_path ==', out_path)

        logger.log(f"Saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    # Synchronize all processes before exiting
    dist.barrier()
    logger.log("Sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,  # Default, can be overridden via CLI
        batch_size=16,      # Default, can be overridden via CLI
        use_ddim=True,
        model_path="",
        SAMPLE_OUT="./samples",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
