"""Switchable MLX compute for the SAM-3D-Body DINOv3 backbone (macOS only).

Only the selected backend sits on the GPU: while MLX serves the backbone the
torch weights are moved to the host (pageable, not wired), and switching back
just moves them to the device again. Decoding, MHR and every other stage keep
running in torch either way.
"""

import contextlib
import importlib
import subprocess
import time

BACKENDS = ("torch", "mlx")
MISSING_MLX = ("需要 MLX 才能使用该骨干后端（仅 macOS）。安装："
               ".venv/bin/pip install -r requirements-macos-mlx.txt")


def mlx_available() -> bool:
    """True when the optional MLX dependency can be imported."""
    try:
        importlib.import_module("mlx.core")
    except Exception:
        return False
    return True


def validate_backend(name) -> str:
    if name not in BACKENDS:
        raise ValueError("sam_backbone must be 'torch' or 'mlx'")
    return name


def wired_gb() -> float:
    """Wired (unswappable) memory; GPU buffers are counted here."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith("Pages wired down"):
                return int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1e9
    except Exception:
        pass
    return 0.0


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


class BackboneSwitcher:
    """Serve ``backbone.forward`` from torch or MLX, one device copy at a time.

    ``use("mlx")`` converts the torch weights and moves the torch module to the
    host; ``use("torch")`` moves it back and drops MLX. The forward dispatcher
    stays installed, so the SAM visualization path keeps working after a switch.
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self._original = backbone.forward
        self._mlx = None
        self._mode = "torch"
        self._host_device = None
        self._patched = False

    # ---------------------------------------------------------------- residency
    def _install(self) -> None:
        if not self._patched:
            self.backbone.forward = self._dispatch
            self._patched = True

    def _dispatch(self, x, extra_embed=None):
        if self._mode == "mlx":
            return self._mlx_forward(x, extra_embed)
        return self._original(x, extra_embed=extra_embed)

    def _mlx_forward(self, x, extra_embed=None):
        if extra_embed is not None:
            raise NotImplementedError("The MLX backbone does not take extra_embed")
        if self._mlx is None:
            raise RuntimeError("MLX backbone missing while torch weights are on the host")
        import numpy as np
        import torch

        host = np.ascontiguousarray(x.detach().cpu().numpy()).astype(np.float16, copy=False)
        return torch.from_numpy(self._mlx(host)).to(device=x.device, dtype=x.dtype)

    def _build_mlx(self):
        if self._mlx is None:
            from .mlx_dinov3 import build_backbone

            t0 = time.time()
            self._mlx = build_backbone(self.backbone)
            print(f"  [sam] MLX backbone ready in {time.time() - t0:.1f}s (FP16)")
        return self._mlx

    def _drop_mlx(self) -> None:
        if self._mlx is not None:
            self._mlx = None
            _clear_mlx_cache()

    def _to_host(self) -> None:
        if self._host_device is not None:
            return
        param = next(self.backbone.parameters())
        self._host_device = param.device
        self.backbone.to("cpu")

    def _to_device(self) -> None:
        if self._host_device is None:
            return
        device, self._host_device = self._host_device, None
        self.backbone.to(device)

    # ------------------------------------------------------------------ switch
    @contextlib.contextmanager
    def use(self, backend):
        backend = validate_backend(backend)
        if backend == "mlx":
            if not mlx_available():
                raise RuntimeError(MISSING_MLX)
            if self._mode != "mlx" or self._mlx is None:
                self._install()
                self._to_device()      # conversion reads the torch weights
                self._build_mlx()
                self._to_host()        # free device memory, keep the host copy
                self._mode = "mlx"
                print(f"  [sam] backbone backend=mlx (torch weights on host, "
                      f"wired {wired_gb():.1f} GB)")
        elif self._mode != "torch":
            self._mode = "torch"
            self._to_device()
            self._drop_mlx()
            print(f"  [sam] backbone backend=torch (MLX copy dropped, "
                  f"wired {wired_gb():.1f} GB)")
        yield self
