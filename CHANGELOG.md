# Changelog — gpu-doctor

> **Verification policy:** No fabricated numbers. Every claim tested on real hardware.

Auto-updated by GitHub Actions on every push to `main`.

---

## [Unreleased] — updated 2026-03-29

### Changed
- Add neon banner (`fc8bae1`)


## [1.0.0] — 2026-03-28

### Added

- `gpu_doctor.detect.get_best_device()` — unified backend detection: DirectML → ROCm → CUDA → MPS → CPU
- `gpu_doctor.detect.get_torch_device()` — returns correct device object for `.to()` calls (DML object, not string)
- `gpu_doctor.detect.get_dtype()` — safe dtype per backend (float32 for DirectML/CPU, float16 elsewhere)
- `gpu_doctor.detect.get_jax_backend()` — JAX backend detection (gpu/cpu)
- `gpu_doctor.detect.configure_jax_amd()` — sets all AMD env vars before JAX import
- `gpu_doctor.detect.device_info()` — full diagnostic dict, JSON-serializable
- `gpu_doctor.detect.check_env()` — human-readable environment report
- `GFX_OVERRIDE_MAP` — auto-applies `HSA_OVERRIDE_GFX_VERSION` for gfx1010/1011/1012/906/900
- CLI: `python -m gpu_doctor --check / --install / --install-jax / --json`
- `pyproject.toml` — PyPI-ready, zero required deps, optional: `[torch]`, `[jax]`, `[all]`
- `README.md` — comparison table vs devicetorch/torchruntime/GPUtil/pyamdgpuinfo
- Verified on: Linux CPU (0.6.2 JAX, 2.11.0 torch), test suite passes

---

_Auto-updated by GitHub Actions changelog workflow._
