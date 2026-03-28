"""
gpu-doctor CLI — run with: python -m gpu_doctor

Usage:
    python -m gpu_doctor                 # detect best device, print summary
    python -m gpu_doctor --check         # full environment report
    python -m gpu_doctor --install       # install correct torch for this machine
    python -m gpu_doctor --install-jax   # install correct jax for this machine
    python -m gpu_doctor --json          # machine-readable JSON output
"""

import argparse
import json
import os
import platform
import subprocess
import sys

OS = platform.system()
PYTHON = sys.executable
PY_VER = sys.version_info[:2]


def install_torch():
    """Install the right torch variant for this machine."""
    from gpu_doctor.detect import _detect_rocm_gfx, GFX_OVERRIDE_MAP

    print("Detecting hardware...")

    if OS == "Windows":
        if PY_VER > (3, 11):
            print(f"[WARN] Python {PY_VER[0]}.{PY_VER[1]} detected.")
            print("       torch-directml requires Python ≤ 3.11 (hard ABI ceiling).")
            print("       For DirectML: create a 3.11 venv first.")
            print("         py -3.11 -m venv .venv311")
            print("         .venv311\\Scripts\\activate")
            print("         python -m gpu_doctor --install")
            print()
            print("       Installing CPU-only torch for current Python...")
            _pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        else:
            print("Installing torch-directml (Windows AMD/Intel/NVIDIA GPU)...")
            print("Note: Do NOT pre-install torch — DirectML pulls torch 2.4.1 automatically.")
            _pip("torch-directml")

    elif OS == "Linux":
        import shutil
        if shutil.which("rocminfo"):
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)
            if "gfx" in result.stdout:
                gfx_ver = _detect_rocm_gfx()
                override = GFX_OVERRIDE_MAP.get(gfx_ver or "")
                print(f"ROCm detected. GPU arch: {gfx_ver}")
                if override:
                    print(f"Note: HSA_OVERRIDE_GFX_VERSION={override} needed for {gfx_ver}")
                    print("      Add to ~/.bashrc: export HSA_OVERRIDE_GFX_VERSION=" + override)
                print("Installing torch for ROCm 6.1...")
                _pip("torch", "--index-url", "https://download.pytorch.org/whl/rocm6.1")
                return
        if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:
            print("NVIDIA GPU detected. Installing torch+CUDA 12.1...")
            _pip("torch", "--index-url", "https://download.pytorch.org/whl/cu121")
            return
        print("No GPU detected. Installing CPU-only torch...")
        _pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    elif OS == "Darwin":
        print("macOS detected. Installing torch (includes MPS for Apple Silicon)...")
        _pip("torch")

    # Verify
    from gpu_doctor.detect import check_env
    check_env()


def install_jax():
    """Install the right JAX variant for this machine."""
    import shutil

    if OS == "Windows":
        print("JAX on Windows: native GPU is not supported.")
        print("For AMD/NVIDIA GPU acceleration on Windows, use WSL2.")
        print("Installing CPU-only JAX...")
        _pip("jax")
        return

    if shutil.which("rocminfo"):
        result = subprocess.run(["rocminfo"], capture_output=True, text=True)
        if "gfx" in result.stdout:
            print("ROCm detected. Installing jax[rocm6_1]...")
            subprocess.run([PYTHON, "-m", "pip", "install", "jax[rocm6_1]",
                            "-f", "https://storage.googleapis.com/jax-releases/rocm/jax_rocm_releases.html"],
                           check=True)
            return

    if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0:
        print("NVIDIA detected. Installing jax[cuda12]...")
        subprocess.run([PYTHON, "-m", "pip", "install", "jax[cuda12]",
                        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"],
                       check=True)
        return

    if OS == "Darwin":
        print("macOS: installing CPU JAX (MPS support experimental)...")
    else:
        print("No GPU detected. Installing CPU JAX...")
    _pip("jax")


def _pip(*args: str):
    subprocess.run([PYTHON, "-m", "pip", "install", "--upgrade"] + list(args), check=True)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m gpu_doctor",
        description="Universal GPU setup and diagnostics for PyTorch and JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check",       action="store_true", help="Full environment report")
    parser.add_argument("--install",     action="store_true", help="Install correct torch")
    parser.add_argument("--install-jax", action="store_true", help="Install correct JAX")
    parser.add_argument("--json",        action="store_true", help="Output JSON (for scripts)")
    args = parser.parse_args()

    from gpu_doctor.detect import check_env, device_info, get_best_device

    if args.json:
        info = device_info()
        print(json.dumps(info, indent=2, default=str))
    elif args.check:
        check_env()
    elif args.install:
        install_torch()
    elif args.install_jax:
        install_jax()
    else:
        # Default: quick summary
        info = device_info()
        best = info.get("best_device", "cpu")
        tv   = info.get("torch_version", "not installed")
        print(f"gpu-doctor  →  best device: {best}  |  torch: {tv}")
        print(f"Run: python -m gpu_doctor --check   for full report")
        print(f"Run: python -m gpu_doctor --install  to install torch")


if __name__ == "__main__":
    main()
