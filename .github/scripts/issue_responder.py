"""
issue_responder.py — Auto-respond to common gpu-doctor issues.

Triggered by GitHub Actions on: issues.opened, issue_comment.created

Pattern matching is intentionally broad — better to respond helpfully to
an edge case than to miss a user who is stuck.
"""

import os
import re
import json
import urllib.request
import urllib.error

# ── Config ──────────────────────────────────────────────────────────────────

REPO = os.environ.get("GITHUB_REPOSITORY", "ChharithOeun/gpu-doctor")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER", "")
ISSUE_TITLE = (os.environ.get("ISSUE_TITLE", "") or "").lower()
ISSUE_BODY = (os.environ.get("ISSUE_BODY", "") or "").lower()
COMMENT_BODY = (os.environ.get("COMMENT_BODY", "") or "").lower()
EVENT_NAME = os.environ.get("EVENT_NAME", "issues")

# Combine title + body + comment for matching
FULL_TEXT = f"{ISSUE_TITLE} {ISSUE_BODY} {COMMENT_BODY}"

# ── Response templates ───────────────────────────────────────────────────────

RESPONSES = [
    # ── No GPU / cuda.is_available() False ──────────────────────────────────
    {
        "triggers": [
            r"cuda.*is_available.*false",
            r"is_available.*false",
            r"no gpu",
            r"gpu not found",
            r"not detecting",
            r"not detected",
            r"can.t find gpu",
            r"cannot find gpu",
            r"gpu not working",
        ],
        "reply": """\
Thanks for opening this! This is almost always fixable.

**Step 1 — Run the diagnostic:**
```bash
python -m gpu_doctor --check
```
This shows exactly what gpu-doctor sees: your OS, Python version, torch build, and which GPU backends are available.

**Step 2 — Common causes:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| AMD GPU on Linux, `cuda.is_available()` → False | gfx1010/gfx906 not in ROCm allow-list | Call `get_best_device()` **before** `import torch` — sets `HSA_OVERRIDE_GFX_VERSION` automatically |
| AMD GPU on Windows | Need DirectML, not ROCm | Python 3.11 required: `py -3.11 -m venv .venv311`, then `pip install torch-directml` |
| NVIDIA GPU not detected | Wrong torch build (CPU wheel installed) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `privateuseone:0` device | DirectML device string bug | Use `get_torch_device()` — returns the device **object**, not the string |

**Step 3 — Minimal repro:**
```python
from gpu_doctor import get_best_device, get_torch_device
print(get_best_device())      # what backend was detected?
print(get_torch_device())     # what device object to use?
```

Paste the output of `--check` and the above snippet here and I'll help narrow it down.
""",
    },

    # ── HSA_OVERRIDE / gfx1010 / RX 5700 XT ────────────────────────────────
    {
        "triggers": [
            r"hsa_override",
            r"gfx1010",
            r"gfx1011",
            r"gfx1012",
            r"gfx906",
            r"gfx900",
            r"rx 5700",
            r"rx 5600",
            r"radeon vii",
            r"vega 56",
            r"vega 64",
            r"rocm.*not.*support",
            r"not in.*allow.?list",
        ],
        "reply": """\
This is the classic ROCm gfx architecture override issue — gpu-doctor handles it automatically.

**The problem:** ROCm's default hardware allow-list doesn't include older RDNA1/Vega cards (gfx1010, gfx906, gfx900). Without an override, `torch.cuda.is_available()` returns False with **no error message**.

**The fix — call `get_best_device()` before importing torch:**
```python
from gpu_doctor import get_best_device   # sets HSA_OVERRIDE_GFX_VERSION automatically
import torch                             # now sees the GPU correctly
print(torch.cuda.is_available())         # True
```

**Or set it manually in your shell:**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 5700 XT / RX 5600 XT (gfx1010)
# export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Radeon VII (gfx906)
# export HSA_OVERRIDE_GFX_VERSION=9.0.0  # RX Vega 56/64 (gfx900)
```
Add this to `~/.bashrc` to make it permanent.

**Auto-applied cards:**
| GPU | Architecture | Override |
|-----|-------------|---------|
| RX 5700 XT, RX 5700 | gfx1010 | 10.3.0 |
| RX 5600 XT, RX 5500 XT | gfx1010–1012 | 10.3.0 |
| Radeon VII | gfx906 | 9.0.6 |
| RX Vega 56/64 | gfx900 | 9.0.0 |

Run `python -m gpu_doctor --check` and share the output if you're still stuck.
""",
    },

    # ── DirectML / Windows ──────────────────────────────────────────────────
    {
        "triggers": [
            r"directml",
            r"torch.directml",
            r"torch-directml",
            r"windows.*amd",
            r"amd.*windows",
            r"privateuseone",
            r"diffusers.*directml",
            r"directml.*diffusers",
        ],
        "reply": """\
DirectML setup on Windows has a few tricky requirements — here's the full working setup:

**Requirements:**
- Python **3.11 or lower** (torch-directml is compiled against the 3.11 ABI; 3.12+ silently fails)
- Do **not** pre-install torch before torch-directml (it pins its own compatible version)

**Install steps:**
```bat
:: Create a 3.11 venv
py -3.11 -m venv .venv311
.venv311\Scripts\activate

pip install gpu-doctor
python -m gpu_doctor --install   :: auto-detects DirectML and installs correctly
python -m gpu_doctor --check     :: verify
```

**Use in code:**
```python
from gpu_doctor import get_best_device, get_torch_device, get_dtype

device_type = get_best_device()   # 'directml'
device = get_torch_device()       # DML device OBJECT (not the string)
dtype = get_dtype()               # float32 (DirectML doesn't support float16 matmul)

model = MyModel().to(device).to(dtype)
```

> **Important:** Always use `get_torch_device()` which returns the device **object**. The string `'privateuseone:0'` causes silent CPU fallback in diffusers and other libraries.

Run `python -m gpu_doctor --check` and paste the output if you're still hitting issues.
""",
    },

    # ── JAX ─────────────────────────────────────────────────────────────────
    {
        "triggers": [
            r"jax",
            r"jaxlib",
            r"configure_jax",
            r"get_jax_backend",
            r"xla",
        ],
        "reply": """\
For JAX on AMD, the environment variables must be set **before** JAX initializes its backend.

**Use gpu-doctor to handle this automatically:**
```python
from gpu_doctor import configure_jax_amd, get_jax_backend

configure_jax_amd()   # sets XLA_FLAGS, MIOPEN_USER_DB_PATH, HSA_OVERRIDE, etc.

import jax            # NOW import jax — backend is already configured
print(get_jax_backend())   # 'gpu' or 'cpu'
```

**Install JAX with the correct backend:**
```bash
python -m gpu_doctor --install-jax   # auto-detects ROCm / CUDA / CPU
```

**Common JAX + AMD issues:**
- NaN results → add `XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'`
- JAX sees CPU even with ROCm → `configure_jax_amd()` not called before `import jax`
- Slow first run → XLA compile cache warming (subsequent runs are fast)

Run `python -m gpu_doctor --check` and share the output if you need more help.
""",
    },

    # ── Install errors ───────────────────────────────────────────────────────
    {
        "triggers": [
            r"pip install.*error",
            r"install.*fail",
            r"cannot install",
            r"modulenotfounderror",
            r"importerror",
            r"no module named",
            r"install.*gpu.doctor",
        ],
        "reply": """\
Let's get the install sorted.

**Standard install (zero required deps):**
```bash
pip install gpu-doctor
```

**With torch:**
```bash
pip install "gpu-doctor[torch]"
```

**With JAX:**
```bash
pip install "gpu-doctor[jax]"
```

**With everything:**
```bash
pip install "gpu-doctor[all]"
```

**Then verify:**
```bash
python -m gpu_doctor --check
```

If you're getting an error, please share:
1. The exact error message
2. Output of `python --version`
3. Output of `python -m gpu_doctor --check` (if it gets that far)

That's usually enough to pinpoint the issue.
""",
    },

    # ── Python version issue ─────────────────────────────────────────────────
    {
        "triggers": [
            r"python 3\.12",
            r"python 3\.13",
            r"python.*version",
            r"3\.12.*directml",
            r"directml.*3\.12",
        ],
        "reply": """\
**DirectML requires Python 3.11 or lower.**

`torch-directml` is compiled against the CPython 3.11 ABI. It will not import on Python 3.12+ — and the error can be subtle (no GPU detected, not a clear import error).

**Check your Python version:**
```bash
python --version
```

**Fix — create a 3.11 virtual environment:**
```bat
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install gpu-doctor
python -m gpu_doctor --install
```

If you don't have Python 3.11 installed:
- Download from https://python.org/downloads (choose 3.11.x)
- Or use the Microsoft Store: search "Python 3.11"

For ROCm (Linux AMD), CUDA (NVIDIA), MPS (Apple Silicon), and CPU — any Python 3.8+ works fine.
""",
    },
]

# ── Fallback response ────────────────────────────────────────────────────────

FALLBACK = """\
Thanks for reaching out! I'm the gpu-doctor automated responder.

To help diagnose your issue, please run:
```bash
python -m gpu_doctor --check
```
And paste the output here. It shows your OS, Python version, torch/JAX install, and detected GPU backends — usually enough to spot the issue immediately.

For common setups:
- **Windows AMD** → needs Python 3.11 + DirectML: `py -3.11 -m venv .venv311 && pip install gpu-doctor && python -m gpu_doctor --install`
- **Linux AMD** → ROCm: `pip install gpu-doctor && python -m gpu_doctor --install`
- **NVIDIA** → CUDA: `pip install gpu-doctor && python -m gpu_doctor --install`
- **Apple Silicon** → MPS: `pip install gpu-doctor && python -m gpu_doctor --install`

A maintainer will follow up shortly.
"""

# ── GitHub API helper ────────────────────────────────────────────────────────


def post_comment(issue_number: str, body: str) -> None:
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    data = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.load(resp)
            print(f"Posted comment: {result.get('html_url', 'ok')}")
    except urllib.error.HTTPError as e:
        print(f"HTTP error posting comment: {e.code} {e.reason}")
        raise


# ── Matching logic ───────────────────────────────────────────────────────────


def find_response(text: str) -> str:
    for rule in RESPONSES:
        for pattern in rule["triggers"]:
            if re.search(pattern, text, re.IGNORECASE):
                matched = pattern
                print(f"Matched pattern: {matched!r}")
                return rule["reply"]
    print("No specific pattern matched — using fallback.")
    return FALLBACK


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not ISSUE_NUMBER:
        print("No ISSUE_NUMBER set — nothing to respond to.")
        return

    if not TOKEN:
        print("No GITHUB_TOKEN — cannot post comment.")
        return

    print(f"Event: {EVENT_NAME}")
    print(f"Issue: #{ISSUE_NUMBER}")
    print(f"Title: {ISSUE_TITLE[:80]!r}")

    # Only respond to issue opens and first-time comments
    # (don't pile on in every comment thread)
    if EVENT_NAME == "issue_comment":
        # Check if this is an early comment — only respond in first 3 comments
        url = f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}/comments?per_page=5"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"token {TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                comments = json.load(resp)
                bot_comments = [c for c in comments if c.get("user", {}).get("login") == "github-actions[bot]"]
                if len(bot_comments) >= 1:
                    print("Already responded to this issue — skipping.")
                    return
        except Exception as e:
            print(f"Could not check comment history: {e}")

    reply = find_response(FULL_TEXT)
    post_comment(ISSUE_NUMBER, reply)
    print("Done.")


if __name__ == "__main__":
    main()
