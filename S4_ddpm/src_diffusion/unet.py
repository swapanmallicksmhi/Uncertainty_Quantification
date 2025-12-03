# -*- coding: utf-8 -*-
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
# Initial date 4 March 2025
# V1 8 September 2025
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


# -------------------------
# Cross-Attention helper
# -------------------------
class CrossAttentionMid(nn.Module):
    """
    Cross-attention block that attends from image features (queries) to cond embeddings (keys/values).
    Uses torch.nn.MultiheadAttention with batch_first=True.
    Expects query_dim == key/value_proj_dim (we project cond to match query dim).
    """
    def __init__(self, query_dim, cond_dim, num_heads=4, attn_dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        # Project cond channels to the query_dim
        self.cond_proj = nn.Conv2d(cond_dim, query_dim, kernel_size=1)
        # MultiheadAttention expects (B, N, E) with batch_first=True
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.ln = nn.LayerNorm(query_dim)
        # small feed-forward
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.ff_ln = nn.LayerNorm(query_dim)

    def forward(self, q_feat, cond):
        """
        q_feat: (B, Cq, Hq, Wq)  -> queries
        cond:   (B, Ccond, Hc, Wc) -> conditioning image (ERA5)
        Returns: same shape as q_feat
        """
        B, Cq, Hq, Wq = q_feat.shape

        # Resize cond spatially to something reasonable — use UNet mid resolution aligned to x input
        cond_resized = F.interpolate(cond, size=(Hq, Wq), mode="bilinear", align_corners=False)  # (B, Ccond, Hq, Wq)
        # project cond to same channels as queries
        cond_proj = self.cond_proj(cond_resized)  # (B, Cq, Hq, Wq)

        # Flatten spatial dims to sequence
        q_seq = q_feat.permute(0, 2, 3, 1).reshape(B, Hq * Wq, Cq)          # (B, Nq, Cq)
        kv_seq = cond_proj.permute(0, 2, 3, 1).reshape(B, Hq * Wq, Cq)      # (B, Nk, Cq)

        # MultiheadAttention (batch_first=True): query, key, value
        attn_out, _ = self.mha(query=q_seq, key=kv_seq, value=kv_seq)       # (B, Nq, Cq)

        # Residual + LN + FF
        h = self.ln(q_seq + attn_out)
        ff_out = self.ff(h)
        h = h + ff_out
        h = self.ff_ln(h)

        # reshape back
        h = h.reshape(B, Hq, Wq, Cq).permute(0, 3, 1, 2).contiguous()      # (B, Cq, Hq, Wq)
        return h


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
    Multi-head spatial attention block (self-attention).
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
    UNet model with attention, timestep embedding and optional cross-attention conditioning.
    If cond (ERA5) is provided to forward(..., cond=...), a cross-attention layer is applied
    at the middle of the model. cond is expected shape (B, C_cond, Hc, Wc), typically C_cond==3.
    """
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8),
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
                 num_heads=1, num_heads_upsample=-1, use_scale_shift_norm=False,
                 cond_channels: int = 3, cond_attn_heads: int = 4):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        # store conditioning info
        self.cond_channels = cond_channels  # if None -> no conditioning support
        self.cond_attn_heads = cond_attn_heads

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

        # Cross-attention module (applied after middle block) if conditioning enabled
        if self.cond_channels is not None:
            # cond_proj to convert cond_channels -> model_channels for attention
            self.cond_proj = conv_nd(dims, self.cond_channels, model_channels, 1)
            self.cross_attn_mid = CrossAttentionMid(query_dim=ch, cond_dim=model_channels, num_heads=self.cond_attn_heads)
        else:
            self.cond_proj = None
            self.cross_attn_mid = None

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
                    layers.append(AttentionBlock(ch, num_heads=num_heads_upsample, use_checkpoint=use_checkpoint))
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
        if self.cross_attn_mid is not None:
            self.cross_attn_mid = convert_module_to_f16(self.cross_attn_mid)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        if self.cross_attn_mid is not None:
            self.cross_attn_mid = convert_module_to_f32(self.cross_attn_mid)

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, cond=None):
        """
        Forward pass.
        x: (B, C, H, W) noisy input (carra2 noisy)
        timesteps: (B,) long tensor
        y: optional class labels
        cond: optional conditioning tensor (ERA5) shape (B, C_cond, Hc, Wc)
        """
        assert (y is not None) == (self.num_classes is not None)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # Middle block
        h = self.middle_block(h, emb)

        # Apply cross-attention in the middle if conditioning provided
        if cond is not None and self.cross_attn_mid is not None:
            # cond may have different spatial size; resize to h spatial dims
            # h has shape (B, C, H_mid, W_mid)
            h = h.type(self.inner_dtype)
            # project cond channels to model_channels (via conv) and then cross-attend
            cond_proj = self.cond_proj(cond.type(h.dtype))
            h = h + self.cross_attn_mid(h, cond_proj)

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

        # if conditioning, optionally include conditioned middle
        # (we do not modify get_feature_vectors to include cond processing by default)

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
        # keep old behaviour: concatenates low-res as extra channels
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        upsampled = F.interpolate(low_res, x.shape[-2:], mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        # If you want to use cross-attn conditioning instead of concatenation,
        # pass `cond=low_res` in kwargs and make SuperResModel.__init__ call UNetModel with cond_channels set.
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        upsampled = F.interpolate(low_res, x.shape[-2:], mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
