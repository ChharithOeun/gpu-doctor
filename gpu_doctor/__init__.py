"""
gpu-doctor — Universal GPU setup, detection, and diagnostics for PyTorch and JAX.

One tool. Every backend. Every OS.

  DirectML (Windows AMD/Intel/NVIDIA)
  ROCm    (Linux/WSL2 AMD)
  CUDA    (Linux/Windows NVIDIA)
  MPS     (macOS Apple Silicon)
  CPU     (any OS, always works)

Quick start:
    pip install gpu-doctor
    python -m gpu_doctor               # detect + print best device
    python -m gpu_doctor --check       # full environment report
    python -m gpu_doctor --install     # install right torch for your hardware

PyTorch users:
    from gpu_doctor import get_best_device, get_torch_device, get_dtype

    device = get_torch_device()        # torch.device, ready for model.to()
    dtype  = get_dtype()               # float16 or float32, safe for your backend

JAX users:
    from gpu_doctor import get_jax_backend, configure_jax_amd

    backend = get_jax_backend()        # 'rocm' | 'cuda' | 'cpu'
    configure_jax_amd()                # sets XLA_FLAGS, HSA_OVERRIDE, etc.
"""

from gpu_doctor.detect import (
    get_best_device,
    get_torch_device,
    get_dtype,
    get_jax_backend,
    configure_jax_amd,
    device_info,
    check_env,
)

__version__ = "1.0.0"
__all__ = [
    "get_best_device",
    "get_torch_device",
    "get_dtype",
    "get_jax_backend",
    "configure_jax_amd",
    "device_info",
    "check_env",
]
