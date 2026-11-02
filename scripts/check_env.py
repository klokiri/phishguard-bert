"""
Environment check for PhishGuard BERT.

Run:  .venv/bin/python scripts/check_env.py

What it does and why:
  1. Prints Python and library versions, so a results file can always be
     traced back to the exact software that produced it.
  2. Reports which accelerator PyTorch can use. This project was planned for
     an NVIDIA GPU (CUDA) but runs on an Apple M4 Pro, where PyTorch uses
     Apple's Metal backend, called "MPS". We print both so the difference is
     visible rather than assumed.
  3. Runs a tiny matrix multiplication on the chosen device, because
     "is_available() == True" does not prove that computation actually works.
  4. Tests which reduced-precision number formats (float16, bfloat16) work on
     the device. Mixed precision makes training faster and uses less memory,
     but support differs between CUDA and MPS, so we measure it here.

Exit code is 0 if a GPU-type device (CUDA or MPS) works, 1 otherwise.
"""

import importlib
import importlib.metadata
import platform
import sys

# Libraries the project depends on. "import name" -> label to print.
# Note: pyquery must be importable because the reference features.py imports
# it at module level, even if we only use its URL/text features.
LIBRARIES = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "pyquery": "pyquery",
    "flask": "flask",
    "flask_cors": "flask-cors",
    "pytest": "pytest",
    "pytest_cov": "pytest-cov",
}


def print_versions() -> bool:
    """Import every library and print its version. Returns False if any fail."""
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    all_ok = True
    for module_name, label in LIBRARIES.items():
        try:
            # Importing proves the library (and any system library it needs,
            # e.g. libomp for xgboost) actually loads, not just that it's installed.
            importlib.import_module(module_name)
            # The version comes from the installed package metadata, which works
            # for every package (some, like pyquery, have no __version__).
            version = importlib.metadata.version(label)
            print(f"{label:<16}: {version}")
        except Exception as exc:  # e.g. ImportError, or a missing system library
            print(f"{label:<16}: FAILED to import -> {exc}")
            all_ok = False
    return all_ok


def pick_device():
    """Return the best available torch device: CUDA, then MPS, then CPU."""
    import torch

    print(f"\ntorch.cuda.is_available()         : {torch.cuda.is_available()}")
    print(f"torch.backends.mps.is_built()     : {torch.backends.mps.is_built()}")
    print(f"torch.backends.mps.is_available() : {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU (CUDA)  : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("GPU (MPS)   : Apple Silicon GPU via Metal")
        return torch.device("mps")
    print("GPU         : none found, falling back to CPU")
    return torch.device("cpu")


def smoke_test(device) -> bool:
    """Multiply two matrices on the device and compare with the CPU answer."""
    import torch

    torch.manual_seed(42)
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    expected = a @ b  # computed on CPU
    result = (a.to(device) @ b.to(device)).cpu()
    ok = torch.allclose(result, expected, atol=1e-3)
    print(f"\nMatmul on {device.type}: {'OK' if ok else 'MISMATCH'}")
    return ok


def precision_test(device) -> None:
    """Try float16 and bfloat16 autocast on the device and report what works.

    autocast = PyTorch automatically runs suitable operations in the smaller
    format. We try a small linear layer forward + backward pass, which is the
    kind of work training does.
    """
    import torch

    print("\nMixed-precision support:")
    for dtype in (torch.float16, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        try:
            layer = torch.nn.Linear(64, 2).to(device)
            x = torch.randn(8, 64, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = layer(x)
            out.float().sum().backward()  # check gradients can flow too
            print(f"  {name:<9}: works (autocast output dtype = {out.dtype})")
        except Exception as exc:
            print(f"  {name:<9}: NOT supported -> {type(exc).__name__}: {exc}")


def main() -> int:
    libs_ok = print_versions()
    device = pick_device()
    math_ok = smoke_test(device)
    precision_test(device)

    gpu_ok = device.type in ("cuda", "mps") and math_ok
    print("\nSUMMARY")
    print(f"  all libraries import : {libs_ok}")
    print(f"  device used          : {device.type}")
    print(f"  GPU computation works: {gpu_ok}")
    return 0 if (libs_ok and gpu_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
