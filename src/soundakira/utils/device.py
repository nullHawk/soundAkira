from __future__ import annotations

import gc
import sys


def resolve_device(spec: str) -> str:
    """'auto' -> 'cuda' if available, else 'mps' on Apple silicon, else 'cpu'."""
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_cuda_device(device: str) -> tuple[str, int]:
    """'cuda:1' -> ('cuda', 1); anything else -> (device, 0)."""
    if device.startswith("cuda"):
        _, _, idx = device.partition(":")
        return "cuda", int(idx or 0)
    return device, 0


def free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
