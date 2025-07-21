"""
Schedule Sampler Classes for Diffusion Models

Implements uniform and loss-aware timestep sampling strategies.
Supports distributed training via torch.distributed for loss synchronization.

Author: Swapan Mallick, SMHI
"""

from abc import ABC, abstractmethod
import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name: str, diffusion) -> "ScheduleSampler":
    """
    Factory function to create a schedule sampler based on name.

    Args:
        name (str): Name of the sampler ('uniform' or 'loss-second-moment').
        diffusion: A diffusion model instance with `num_timesteps` defined.

    Returns:
        ScheduleSampler: An instance of a schedule sampler.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"Unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for sampling timesteps during training.
    """

    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Array of weights for each timestep (not necessarily normalized).
        """
        pass

    def sample(self, batch_size: int, device: th.device):
        """
        Importance-sample timesteps for a batch using the current weight distribution.

        Args:
            batch_size (int): Number of samples to draw.
            device (torch.device): Device on which to return tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - indices: Sampled timestep indices.
                - weights: Associated importance weights.
        """
        w = self.weights()
        p = w / np.sum(w)

        indices_np = np.random.choice(len(p), size=batch_size, p=p)
        indices = th.from_numpy(indices_np).long().to(device)

        weights_np = 1.0 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)

        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniformly sample timesteps from the diffusion process.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps], dtype=np.float64)

    def weights(self) -> np.ndarray:
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Base class for samplers that adapt based on per-timestep training losses.
    Supports distributed synchronization.
    """

    def update_with_local_losses(self, local_ts: th.Tensor, local_losses: th.Tensor):
        """
        Gather local timestep losses across distributed processes.

        Args:
            local_ts (Tensor): Local timestep indices.
            local_losses (Tensor): Corresponding loss values.
        """
        world_size = dist.get_world_size()
        device = local_ts.device

        # Gather batch sizes from all ranks
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=device) for _ in range(world_size)
        ]
        dist.all_gather(batch_sizes, th.tensor([len(local_ts)], dtype=th.int32, device=device))
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        # Gather and pad data
        timestep_batches = [th.zeros(max_bs, device=device) for _ in range(world_size)]
        loss_batches = [th.zeros(max_bs, device=device) for _ in range(world_size)]

        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Unpad and combine results
        all_timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        all_losses = [
            x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]
        ]

        self.update_with_all_losses(all_timesteps, all_losses)

    @abstractmethod
    def update_with_all_losses(self, ts: list, losses: list):
        """
        Update internal state using all gathered timestep-loss pairs.

        Args:
            ts (list): List of timestep indices.
            losses (list): Corresponding loss values.
        """
        pass


class LossSecondMomentResampler(LossAwareSampler):
    """
    Resample timesteps based on the second moment (variance) of loss per step.
    """

    def __init__(self, diffusion, history_per_term: int = 10, uniform_prob: float = 0.001):
        """
        Args:
            diffusion: The diffusion model.
            history_per_term (int): Number of past loss values to track per timestep.
            uniform_prob (float): Probability of uniform sampling to maintain exploration.
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        self._loss_history = np.zeros(
            (diffusion.num_timesteps, history_per_term), dtype=np.float64
        )
        self._loss_counts = np.zeros(diffusion.num_timesteps, dtype=np.int32)

    def weights(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Importance weights for each timestep.
        """
        if not self._warmed_up():
            return np.ones(self.diffusion.num_timesteps, dtype=np.float64)

        # Compute second moment (variance proxy)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=1))

        # Normalize and apply uniform mixing
        weights /= np.sum(weights)
        weights = weights * (1.0 - self.uniform_prob)
        weights += self.uniform_prob / len(weights)

        return weights

    def update_with_all_losses(self, ts: list, losses: list):
        """
        Store new loss values in history buffer.

        Args:
            ts (list): Timestep indices.
            losses (list): Corresponding loss values.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] < self.history_per_term:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1
            else:
                # Shift left to make room for new entry
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss

    def _warmed_up(self) -> bool:
        """
        Returns:
            bool: True if all timesteps have filled their history buffers.
        """
        return np.all(self._loss_counts == self.history_per_term)
