"""Verify a macOS FlowHMR install: PyTorch backend, model files, optional extras."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))


def main():
    import torch

    os.chdir(ROOT)
    result = {"python": sys.version.split()[0], "machine": os.uname().machine, "errors": [], "optional": {}}
    result["torch"] = torch.__version__
    mps = getattr(torch.backends, "mps", None)
    result["mps"] = bool(mps and mps.is_available())
    device = "mps" if result["mps"] else "cpu"
    result["device"] = device
    for name in ["torchvision", "numpy", "cv2", "flask", "yolox", "supervision", "motion_correction",
                 "rgbmocap_correction", "flowhmr.pipeline.v2m_inference",
                 "flowhmr.utils.runtime_sam_features", "flowhmr.utils.runtime_vggt_camera"]:
        try:
            importlib.import_module(name)
        except Exception as error:
            if name in ("motion_correction", "rgbmocap_correction"):
                result["optional"][name] = f"unavailable: {type(error).__name__}: {error}"
            else:
                result["errors"].append(f"{name}: {type(error).__name__}: {error}")
    try:
        x = torch.randn(64, 64, device=device)
        assert torch.isfinite(x @ x).all().item()
        result["matmul"] = "passed"
    except Exception as error:
        result["errors"].append(f"{device} matmul: {type(error).__name__}: {error}")
    result["ffmpeg"] = shutil.which("ffmpeg")
    if result["ffmpeg"] is None:
        try:
            import imageio_ffmpeg

            result["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as error:
            result["errors"].append(f"FFmpeg: {error}")

    from flowhmr import model_integrity
    from flowhmr.model_files import MODEL_ROOT, core_model_paths, smplh_reference

    result["body_reference"] = smplh_reference()
    result["models"] = model_integrity.summary(deep=False)
    result["model_directory"] = str(MODEL_ROOT)
    result["missing_models"] = [item["file"] for item in model_integrity.verify(required_paths=core_model_paths()) if not item.ok]
    result["ready"] = (not result["errors"] and not result["missing_models"]
                       and result["models"]["ok"] == result["models"]["checked"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
