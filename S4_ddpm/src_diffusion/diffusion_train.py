"""
Author: Swapan Mallick
Date : 10 March 2025

Key concept:
------------
bucket_cap_mb=128
- 
"""
import os
import copy
import functools

import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import diffusion_dist, logger
from .diffusion_fp16 import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = (
            [self._load_ema_parameters(rate) for rate in self.ema_rate]
            if self.resume_step
            else [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        )

        self.use_ddp = th.cuda.is_available()
        self.ddp_model = (
            DDP(
                self.model,
                device_ids=[diffusion_dist.dev()],
                output_device=diffusion_dist.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            if self.use_ddp
            else self.model
        )
        if not self.use_ddp and dist.get_world_size() > 1:
            logger.warn("Distributed training requires CUDA. Gradients will not sync correctly!")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    diffusion_dist.load_state_dict(resume_checkpoint, map_location=diffusion_dist.dev())
                )
        diffusion_dist.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(checkpoint, self.resume_step, rate)
        if ema_checkpoint and dist.get_rank() == 0:
            logger.log(f"Loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = diffusion_dist.load_state_dict(ema_checkpoint, map_location=diffusion_dist.dev())
            ema_params = self._state_dict_to_master_params(state_dict)
        diffusion_dist.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = diffusion_dist.load_state_dict(opt_checkpoint, map_location=diffusion_dist.dev())
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while not self.lr_anneal_steps or (self.step + self.resume_step) < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.getenv("DIFFUSION_TRAINING_TEST") and self.step > 0:
                    return
            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_fp16() if self.use_fp16 else self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(diffusion_dist.dev())
            micro_cond = {k: v[i:i + self.microbatch].to(diffusion_dist.dev()) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], diffusion_dist.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            losses = compute_losses() if last_batch or not self.use_ddp else self.ddp_model.no_sync().__enter__() or compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            (loss * 2 ** self.lg_loss_scale).backward() if self.use_fp16 else loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = sum((p.grad ** 2).sum().item() for p in self.master_params)
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if self.lr_anneal_steps:
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                filename = (
                    f"model{self.step+self.resume_step:06d}.pt" if not rate
                    else f"ema_{rate}_{self.step+self.resume_step:06d}.pt"
                )
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt{self.step+self.resume_step:06d}.pt"), "wb") as f:
                th.save(self.opt.state_dict(), f)
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(self.model.parameters(), master_params)
        state_dict = self.model.state_dict()
        for i, (name, _) in enumerate(self.model.named_parameters()):
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return make_master_params(params) if self.use_fp16 else params


def parse_resume_step_from_filename(filename):
    try:
        return int(filename.split("model")[-1].split(".")[0])
    except (IndexError, ValueError):
        return 0


def get_blob_logdir():
    return os.getenv("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if not main_checkpoint:
        return None
    path = bf.join(bf.dirname(main_checkpoint), f"ema_{rate}_{step:06d}.pt")
    return path if bf.exists(path) else None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for t_idx, loss_val in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * t_idx / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", loss_val)
