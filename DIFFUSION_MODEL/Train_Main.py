"""
Main program for training diffusion models on image data.
For Uncetainty Quantification Over the CARRA2 domain

Author: Swapan Mallick
Date : 9 March 2025

Key concept:
------------
- ema_rate:
    Exponential Moving Average (EMA) rate is used to maintain a smoothed copy of model weights.
    This stabilizes training by reducing noise and helps with better generalization.
"""

import argparse
import sys
import torch as th

# Add the project root and current directory to path to allow imports
sys.path.append("..")
sys.path.append(".")

# Import custom modules from the diffusion training framework
from src_diffusion import diffusion_dist, logger
from src_diffusion.image_datasets import load_data
from src_diffusion.resample import create_named_schedule_sampler
from src_diffusion.diffusion_script import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src_diffusion.diffusion_train import TrainLoop

def create_argparser():
    """
    Creates the command-line argument parser with default training configurations.
    """
    defaults = dict(
        data_dir="",                 # Directory containing training images
        TRAIN_OUT="",                # Directory containing training images
        schedule_sampler="uniform", # Sampling schedule for time steps
        lr=1e-4,                     # Learning rate
        weight_decay=0.0,           # Optional L2 regularization
        lr_anneal_steps=0,          # Steps for learning rate annealing (0 disables)
        batch_size=1,               # Number of samples per batch
        microbatch=-1,              # Microbatching (-1 disables)
        ema_rate="0.9999",          # EMA decay rate for model parameters
        log_interval=1000,             # Logging interval (in training steps)
        save_interval=1000,           # How often to save model checkpoints
        resume_checkpoint="",       # Path to resume training from a checkpoint
        use_fp16=False,             # Use mixed precision training (for speed and memory)
        fp16_scale_growth=1e-3,     # FP16 scaling factor growth
    )

    # Add model and diffusion-specific defaults
    defaults.update(model_and_diffusion_defaults())

    # Create parser and add arguments
    parser = argparse.ArgumentParser(description="Train a diffusion model on images")
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    """
    The main training entry point.
    Sets up distributed training, loads data/model, and runs the training loop.
    """
    args = create_argparser().parse_args()

    # Initialize distributed computing (e.g., multi-GPU)
    diffusion_dist.setup_dist()

    # Initialize logging utility
    #logger.configure()
    logger.configure(dir=args.TRAIN_OUT)
    logger.log("Creating model and diffusion process...")

    # Instantiate the model and the diffusion object
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Move model to the correct device (e.g., GPU)
    model.to(diffusion_dist.dev())

    # Create the schedule sampler for selecting timesteps
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Loading dataset...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,  # Whether to use class-conditional training
    )

    logger.log("Starting training loop...")

    # Run the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
