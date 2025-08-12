"""
Schedule Sampler and Super-Resolution Model

This module implements:
- A sampling strategy for training diffusion models, including:
    - Uniform timestep sampling
    - Loss-aware sampling (using second-moment loss statistics)
    - Distributed training for loss-aware sampling
- A UNet-based super-resolution model that uses a low-resolution image
  as a conditional input.

Author: Swapan Mallick, SMHI
"""

import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_fp16 import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


# ============================ #
#      Core Base Modules      #
# ============================ #

class TimestepBlock(nn.Module):
    """
    Base class for any module that accepts timestep embeddings.
    """
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential block that supports timestep embeddings.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsample layer with optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    """
    Downsample layer with optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1) if use_conv else avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ============================ #
#         ResNet Block        #
# ============================ #

class ResBlock(TimestepBlock):
    """
    Residual block with optional scale/shift normalization and checkpointing.
    """
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            kernel_size = 3 if use_conv else 1
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=1 if use_conv else 0)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# ============================ #
#        Attention Block       #
# ============================ #

class AttentionBlock(nn.Module):
    """
    Multi-head spatial attention block.
    """
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)).reshape(b * self.num_heads, -1, x.shape[-1])
        h = self.attention(qkv).reshape(b, c, -1)
        return (x + self.proj_out(h)).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Query-Key-Value attention mechanism.
    """
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)


# ============================ #
#         UNet Model           #
# ============================ #

class UNetModel(nn.Module):
    """
    UNet model with attention and timestep embedding.
    """
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8),
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
                 num_heads=1, num_heads_upsample=-1, use_scale_shift_norm=False):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1)))
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Downsample path
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        # Middle
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm)
        )

        # Upsample path
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(), time_embed_dim, dropout,
                        out_channels=model_channels * mult, dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads_upsample, use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1))
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (self.num_classes is not None)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb += self.label_emb(y)
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(th.cat([h, hs.pop()], dim=1), emb)
        return self.out(h.type(x.dtype))

    def get_feature_vectors(self, x, timesteps, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb += self.label_emb(y)

        h = x.type(self.inner_dtype)
        result = {'down': [], 'up': []}
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result['down'].append(h.type(x.dtype))

        h = self.middle_block(h, emb)
        result['middle'] = h.type(x.dtype)

        for module in self.output_blocks:
            h = module(th.cat([h, hs.pop()], dim=1), emb)
            result['up'].append(h.type(x.dtype))

        return result


# ============================ #
#     Super-Resolution Net     #
# ============================ #

class SuperResModel(UNetModel):
    """
    UNetModel for super-resolution conditioned on a low-res image.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        upsampled = F.interpolate(low_res, x.shape[-2:], mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        upsampled = F.interpolate(low_res, x.shape[-2:], mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
