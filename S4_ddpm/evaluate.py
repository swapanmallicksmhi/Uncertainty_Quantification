#!/usr/bin/env python3
"""
===============================================================================
 Program: Evaluation Script - Single Best Quality Image (Modified)
 Author:  Swapan Mallick, SMHI
 Date:    2025-09-06
===============================================================================
Main entry point for the evaluation pipeline.
"""

import os
import sys
import argparse
import traceback
import torch.distributed as dist
from src_diffusion.evaluate_utils import setup_rank_logging, save_error_trace, init_distributed_from_env
from src_diffusion.evaluate_process import evaluate_process

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
            evaluate_process(rank_for_log if rank is not None else 0, world_size, args, LOG)
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
