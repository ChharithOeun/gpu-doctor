"""
gpu_doctor.detect — Core detection logic.

Detection priority (matches real-world performance for each platform):
  Windows : DirectML > CUDA > CPU
  Linux   : CUDA > ROCm > CPU
  macOS   : MPS > CPU

HSA_OVERRIDE_GFX_VERSION is set automatically for known unsupported GPUs
(gfx1010 family — RX 5700 XT, RX 5700, RX 5600 XT) before torch imports.
This is the #1 reason AMD GPU users get silent CPU fallback on ROCm.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Optional

OS = platform.system()   # 'Windows', 'Linux', 'Darwin'

# ── GFX override map ─────────────────────────────────────────────────────────
# Maps GPU architecture IDs to the HSA_OVERRIDE_GFX_VERSION needed for ROCm.
# These GPUs are NOT in ROCm's default hardware allow-list.
GFX_OVERRIDE_MAP = {
    "gfx1010": "10.3.0",   # Navi 10: RX 5700 XT, RX 5700, RX 5600 XT, RX 5500 XT
    "gfx1011": "10.3.0",   # Navi 12: Radeon Pro 5600M
    "gfx1012": "10.3.0",   # Navi 14: RX 5300, RX 5300M
    "gfx906":  "9.0.6",    # Vega 20: Radeon VII, Instinct MI50/MI60
    "gfx900":  "9.0.0",    # Vega 10: RX Vega 56/64
}

# ── Detect GPU arch from rocminfo ─────────────────────────────────────────────
def _detect_rocm_gfx() -> Optional[str]:
    """Return gfx architecture string from rocminfo, e.g. 'gfx1010'."""
    if not shutil.which("rocminfo"):
        return None
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        match = re.search(r"(gfx\d+)", out.stdout)
        return match.group(1) if match else None
    except Exception:
        return None


def _apply_gfx_override(gfx: Optional[str]) -> Optional[str]:
    """
    If the detected GPU needs an HSA override, set it in os.environ NOW —
    before torch or jax imports. Returns the version string set, or None.
    """
    if not gfx:
        return None
    override = GFX_OVERRIDE_MAP.get(gfx)
    if override and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = override
        return override
    return os.environ.get("HSA_OVERRIDE_GFX_VERSION")


# ── Main device detection ─────────────────────────────────────────────────────
def get_best_device() -> str:
    """
    Detect the best available compute device.

    Returns one of: 'directml' | 'cuda' | 'rocm' | 'mps' | 'cpu'

    Side effects:
      - Sets HSA_OVERRIDE_GFX_VERSION if needed for gfx1010-family GPUs.
      - Must be called BEFORE importing torch or jax so env vars take effect.

    Example:
        from gpu_doctor import get_best_device
        device_type = get_best_device()   # 'rocm', 'directml', etc.
    """
    # ── Windows: DirectML first ───────────────────────────────────────────────
    if OS == "Windows":
        try:
            import torch_directml  # noqa: F401
            return "directml"
        except ImportError:
            pass

    # ── ROCm / CUDA (via torch.cuda) ─────────────────────────────────────────
    # Apply GFX override BEFORE importing torch so the env var is in place
    gfx = _detect_rocm_gfx()
    _apply_gfx_override(gfx)

    try:
        import torch
        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            return "rocm" if hip else "cuda"
        # MPS — Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def get_torch_device(device_type: Optional[str] = None):
    """
    Return a device object ready for model.to().

    For DirectML returns the DML device object (NOT the string 'privateuseone:0').
    For all others returns torch.device.

    IMPORTANT: Always use this method, not torch.device('privateuseone:0').
    Passing the string to diffusers .to() silently falls back to CPU.

    Example:
        device = get_torch_device()
        model.to(device)   # works correctly for all backends
    """
    import torch
    if device_type is None:
        device_type = get_best_device()

    if device_type == "directml":
        try:
            import torch_directml as dml
            return dml.device()   # device object, not string
        except ImportError:
            pass
    elif device_type in ("cuda", "rocm"):
        return torch.device("cuda:0")
    elif device_type == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device_type: Optional[str] = None):
    """
    Return the recommended dtype for your backend.

    DirectML float16 is unreliable on most cards — returns float32.
    Everything else returns float16.

    Example:
        dtype = get_dtype()
        model.to(dtype)
    """
    import torch
    if device_type is None:
        device_type = get_best_device()
    if device_type in ("directml", "cpu"):
        return torch.float32
    return torch.float16


# ── JAX ───────────────────────────────────────────────────────────────────────
def get_jax_backend() -> str:
    """
    Return JAX backend string: 'gpu' | 'cpu'.

    Also applies AMD env var workarounds before JAX imports.

    Example:
        from gpu_doctor import get_jax_backend, configure_jax_amd
        configure_jax_amd()      # set XLA_FLAGS, MIOPEN, etc.
        backend = get_jax_backend()
    """
    try:
        import jax
        devs = jax.devices()
        platforms = {d.platform for d in devs}
        if "gpu" in platforms:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def configure_jax_amd(gfx_version: Optional[str] = None):
    """
    Set all required AMD environment variables for JAX + ROCm.
    Must be called BEFORE importing jax.

    Sets:
      XLA_FLAGS                  — disables Triton GEMM (causes NaN on AMD)
      MIOPEN_USER_DB_PATH        — writable MIOpen cache (avoids permission errors)
      JAX_COMPILATION_CACHE_DIR  — XLA compilation cache
      HIP_VISIBLE_DEVICES        — restrict to GPU 0
      HSA_OVERRIDE_GFX_VERSION   — for unsupported GPU architectures (gfx1010, etc.)

    Args:
        gfx_version: Override string like "10.3.0". If None, auto-detects from rocminfo.

    Example:
        from gpu_doctor import configure_jax_amd
        configure_jax_amd()     # auto-detects your GPU
        import jax               # now sees GPU correctly
    """
    # Auto-detect if not provided
    if gfx_version is None:
        gfx = _detect_rocm_gfx()
        gfx_version = GFX_OVERRIDE_MAP.get(gfx or "")

    defaults = {
        "XLA_FLAGS":                 "--xla_gpu_enable_triton_gemm=false",
        "MIOPEN_USER_DB_PATH":       "/tmp/miopen-cache",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/jax-cache",
        "HIP_VISIBLE_DEVICES":       "0",
    }
    if gfx_version:
        defaults["HSA_OVERRIDE_GFX_VERSION"] = gfx_version

    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v

    os.makedirs(os.environ["MIOPEN_USER_DB_PATH"], exist_ok=True)
    os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────
def device_info() -> dict:
    """
    Return a full diagnostic dictionary.

    Useful for logging, bug reports, and --check mode.
    All values are strings or None — safe to serialize to JSON.
    """
    info: dict = {
        "gpu_doctor_version": "1.0.0",
        "python_version":     sys.version.split()[0],
        "platform":           platform.platform(),
        "os":                 OS,
        "machine":            platform.machine(),
    }

    # Best device
    info["best_device"] = get_best_device()

    # GFX override
    gfx = _detect_rocm_gfx()
    info["rocm_gfx_arch"]         = gfx
    info["hsa_override_applied"]  = GFX_OVERRIDE_MAP.get(gfx or "", None)

    # torch
    try:
        import torch
        info["torch_version"]  = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_vram_mb"]     = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            info["rocm_hip_version"] = getattr(torch.version, "hip", None)
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        info["torch_version"] = None

    # DirectML (Windows)
    if OS == "Windows":
        try:
            import torch_directml as dml
            info["directml_version"]  = dml.__version__
            info["directml_gpu_name"] = dml.device_name(0) if hasattr(dml, "device_name") else "unknown"
        except ImportError:
            info["directml_version"] = None

    # JAX
    try:
        import jax
        info["jax_version"]  = jax.__version__
        info["jax_devices"]  = str(jax.devices())
        info["jax_backend"]  = get_jax_backend()
    except ImportError:
        info["jax_version"] = None

    # System tools
    for cmd in ("rocminfo", "rocm-smi", "nvidia-smi"):
        info[f"{cmd.replace('-', '_')}_found"] = bool(shutil.which(cmd))

    return info


def check_env(verbose: bool = True) -> dict:
    """
    Print a human-readable environment report and return device_info dict.

    Example:
        from gpu_doctor import check_env
        info = check_env()
    """
    info = device_info()

    if not verbose:
        return info

    W = 66
    print("=" * W)
    print("  gpu-doctor — Environment Report")
    print("=" * W)
    print(f"  gpu-doctor : v{info.get('gpu_doctor_version', '?')}")
    print(f"  Python     : {info['python_version']}")
    print(f"  OS         : {info['os']}  {info['machine']}")
    print()

    # torch
    tv = info.get("torch_version")
    if tv:
        print(f"  torch      : {tv}")
        if info.get("cuda_available"):
            label = "ROCm" if info.get("rocm_hip_version") else "CUDA"
            print(f"  {label:<11}: {info.get('cuda_device_name')}  "
                  f"({info.get('cuda_vram_mb')} MB VRAM)")
            if info.get("rocm_hip_version"):
                print(f"  HIP        : {info['rocm_hip_version']}")
        else:
            print("  CUDA/ROCm  : not available")
        mps = info.get("mps_available")
        print(f"  MPS        : {'available (Apple Silicon)' if mps else 'not available'}")
        if OS == "Windows":
            dv = info.get("directml_version")
            if dv:
                print(f"  DirectML   : {dv}  ({info.get('directml_gpu_name')})")
            else:
                print("  DirectML   : not installed  (pip install torch-directml, Python ≤ 3.11)")
    else:
        print("  torch      : NOT INSTALLED")
        _print_install_hint()

    # JAX
    jv = info.get("jax_version")
    if jv:
        print(f"  JAX        : {jv}  →  {info.get('jax_backend', '?')}")
    else:
        print("  JAX        : not installed  (pip install jax)")

    # ROCm GFX override
    gfx = info.get("rocm_gfx_arch")
    override = info.get("hsa_override_applied")
    if gfx:
        print(f"  GPU arch   : {gfx}")
        if override:
            print(f"  HSA override: {override}  (auto-applied for {gfx})")
        else:
            print(f"  HSA override: not needed for {gfx}")

    # System tools
    print()
    for cmd in ("rocminfo", "rocm-smi", "nvidia-smi"):
        key = cmd.replace("-", "_") + "_found"
        status = "found" if info.get(key) else "not found"
        print(f"  {cmd:<14}: {status}")

    print()
    best = info.get("best_device", "cpu")
    print(f"  Best device: {best}  ← use this")
    print("=" * W)

    return info


def _print_install_hint():
    hints = {
        "Windows": (
            "  Install torch for Windows:\n"
            "    AMD/Intel/NVIDIA GPU (Python 3.11 only):\n"
            "      pip install torch-directml\n"
            "    NVIDIA CUDA:\n"
            "      pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "    CPU only:\n"
            "      pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ),
        "Linux": (
            "  Install torch for Linux:\n"
            "    AMD ROCm:  pip install torch --index-url https://download.pytorch.org/whl/rocm6.1\n"
            "    NVIDIA:    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "    CPU only:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ),
        "Darwin": (
            "  Install torch for macOS:\n"
            "    pip install torch  (includes MPS for Apple Silicon)"
        ),
    }
    print(hints.get(OS, hints["Linux"]))
