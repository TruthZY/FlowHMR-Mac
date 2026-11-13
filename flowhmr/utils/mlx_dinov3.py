"""MLX implementation of the SAM-3D-Body DINOv3 ViT-H+ backbone.

Mirrors third_party/dinov3 (32 global-attention blocks, axial RoPE on patch
tokens only, SwiGLU FFN, LayerScale, 1 cls + 4 storage tokens). Weights are
converted from the already-loaded PyTorch backbone, so this module never touches
the checkpoint on disk and stays in sync with whatever SAM loaded.

Importing this module requires ``mlx``; it is only imported when the MLX
backend is selected. MLX runs the weights in FP16: on MPS that matches the
PyTorch FP16 backbone to ~2e-3 relative L2 and is ~1.3x faster.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

ROPE_BASE = 100.0


def _np(tensor) -> np.ndarray:
    """FP16 host copy; converting on the host avoids a float32 device copy."""
    return tensor.detach().cpu().numpy().astype(np.float16)


def _f16(array: np.ndarray) -> mx.array:
    return mx.array(array)


def rope_tables(height: int, width: int, head_dim: int) -> tuple[mx.array, mx.array]:
    """Axial RoPE tables, matching RopePositionEmbedding(normalize_coords="separate")."""
    coords_h = np.arange(0.5, height, dtype=np.float32) / np.float32(height)
    coords_w = np.arange(0.5, width, dtype=np.float32) / np.float32(width)
    grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([grid_h, grid_w], axis=-1).reshape(-1, 2) * 2.0 - 1.0
    periods = ROPE_BASE ** (2 * np.arange(head_dim // 4, dtype=np.float32) / (head_dim // 2))
    angles = (2 * np.pi * coords[:, :, None] / periods[None, None, :]).reshape(-1, head_dim // 2)
    angles = np.tile(angles, 2).astype(np.float32)
    return mx.array(np.sin(angles)), mx.array(np.cos(angles))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, params: dict, prefix: str):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm1.weight = _f16(params[prefix + "norm1.weight"])
        self.norm1.bias = _f16(params[prefix + "norm1.bias"])

        self.qkv = nn.Linear(dim, 3 * dim)
        self.qkv.weight = _f16(params[prefix + "attn.qkv.weight"])
        mask = params.get(prefix + "attn.qkv.bias_mask")
        bias = params[prefix + "attn.qkv.bias"]
        self.qkv.bias = _f16(bias * mask if mask is not None else bias)

        self.proj = nn.Linear(dim, dim)
        self.proj.weight = _f16(params[prefix + "attn.proj.weight"])
        self.proj.bias = _f16(params[prefix + "attn.proj.bias"])
        self.gamma1 = _f16(params[prefix + "ls1.gamma"])

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2.weight = _f16(params[prefix + "norm2.weight"])
        self.norm2.bias = _f16(params[prefix + "norm2.bias"])

        for name in ("w1", "w2", "w3"):
            layer = nn.Linear(dim, hidden) if name != "w3" else nn.Linear(hidden, dim)
            layer.weight = _f16(params[prefix + f"mlp.{name}.weight"])
            layer.bias = _f16(params[prefix + f"mlp.{name}.bias"])
            setattr(self, name, layer)
        self.gamma2 = _f16(params[prefix + "ls2.gamma"])

    @staticmethod
    def _rope(x, sin, cos):
        dtype = x.dtype
        x = x.astype(mx.float32)
        x1, x2 = mx.split(x, 2, axis=-1)
        out = x * cos + mx.concatenate([-x2, x1], axis=-1) * sin
        return out.astype(dtype)

    def __call__(self, x, sin, cos, prefix):
        b, n, _ = x.shape
        heads, head_dim = self.heads, self.head_dim
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, heads, head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        # Only patch tokens carry positional information.
        q = mx.concatenate([q[:, :, :prefix], self._rope(q[:, :, prefix:], sin, cos)], axis=2)
        k = mx.concatenate([k[:, :, :prefix], self._rope(k[:, :, prefix:], sin, cos)], axis=2)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, n, self.dim)
        x = x + self.proj(attn) * self.gamma1
        h = self.norm2(x)
        return x + self.w3(nn.silu(self.w1(h)) * self.w2(h)) * self.gamma2


class MlxDinoBackbone(nn.Module):
    """Drop-in replacement for the tokens branch of ``Dinov3Backbone.forward``."""

    def __init__(self, params: dict, *, dim: int, heads: int, depth: int,
                 patch_size: int, storage_tokens: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.patch_size = patch_size
        self.prefix = 1 + storage_tokens
        self.conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.conv.weight = _f16(np.transpose(params.pop("patch_embed.proj.weight"), (0, 2, 3, 1)))
        self.conv.bias = _f16(params.pop("patch_embed.proj.bias"))
        self.cls = _f16(params.pop("cls_token").reshape(1, 1, dim))
        if storage_tokens:
            self.storage = _f16(params.pop("storage_tokens").reshape(1, storage_tokens, dim))
        # Consume each block's weights as it is built so only one block's host
        # copy is alive at a time (the full dict would double the footprint).
        blocks = []
        for index in range(depth):
            prefix = f"blocks.{index}."
            hidden = params[prefix + "mlp.w1.weight"].shape[0]
            blocks.append(Block(dim, heads, hidden, params, prefix))
            for key in [key for key in params if key.startswith(prefix)]:
                del params[key]
        self.blocks = blocks
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.norm.weight = _f16(params.pop("norm.weight"))
        self.norm.bias = _f16(params.pop("norm.bias"))
        params.clear()
        self._rope_cache = {}

    def _rope(self, height, width):
        key = (height, width)
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(height, width, self.dim // self.heads)
        return self._rope_cache[key]

    def __call__(self, x):
        """x: (B, 3, H, W) FP16 array -> (B, dim, H/16, W/16) FP16 numpy array."""
        x = x if isinstance(x, mx.array) else mx.array(x)
        b = x.shape[0]
        tokens = mx.conv2d(x.transpose(0, 2, 3, 1), self.conv.weight,
                           stride=self.patch_size) + self.conv.bias
        grid_h, grid_w = tokens.shape[1], tokens.shape[2]
        patches = tokens.reshape(b, grid_h * grid_w, self.dim)
        cls = mx.broadcast_to(self.cls, (b, 1, self.dim))
        parts = [cls]
        if self.prefix > 1:
            parts.append(mx.broadcast_to(self.storage, (b, self.prefix - 1, self.dim)))
        parts.append(patches)
        tokens = mx.concatenate(parts, axis=1)
        sin, cos = self._rope(grid_h, grid_w)
        for block in self.blocks:
            tokens = block(tokens, sin, cos, self.prefix)
        tokens = self.norm(tokens[:, self.prefix:])
        out = tokens.reshape(b, grid_h, grid_w, self.dim).transpose(0, 3, 1, 2)
        mx.eval(out)
        return np.array(out)


def build_backbone(torch_backbone) -> MlxDinoBackbone:
    """Convert a loaded Dinov3Backbone into the MLX implementation."""
    encoder = torch_backbone.encoder
    state = {key: _np(value) for key, value in encoder.state_dict().items()
             if hasattr(value, "detach")}
    state["cls_token"] = _np(encoder.cls_token)
    state["mask_token"] = _np(encoder.mask_token)
    state["norm.weight"] = _np(encoder.norm.weight)
    state["norm.bias"] = _np(encoder.norm.bias)
    if getattr(encoder, "n_storage_tokens", 0):
        state["storage_tokens"] = _np(encoder.storage_tokens)
    return MlxDinoBackbone(
        state,
        dim=encoder.embed_dim,
        heads=encoder.num_heads,
        depth=encoder.n_blocks,
        patch_size=int(encoder.patch_size),
        storage_tokens=int(getattr(encoder, "n_storage_tokens", 0)),
    )
