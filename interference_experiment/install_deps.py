"""
Install all experiment dependencies into the current environment.
Run once before running run_all.py:

    python install_deps.py

Installs: torch, transformers, scipy, scikit-learn, matplotlib, seaborn, networkx
datasets and numpy are already present in the project venv.
"""

import subprocess
import sys

PACKAGES = [
    # PyTorch — install CUDA 12.x wheel from the official index
    "torch --index-url https://download.pytorch.org/whl/cu121",
    "transformers>=4.36.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "networkx>=3.2",
]

def install(pkg: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pkg.split()
    print(f"  Installing: {pkg.split()[0]} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: {result.stderr.strip()}")
    else:
        print(f"  OK")

if __name__ == "__main__":
    print("=" * 50)
    print("Installing interference experiment dependencies")
    print("=" * 50)
    for pkg in PACKAGES:
        install(pkg)
    print("\nDone.  Verify with:")
    print("  python -c \"import torch, transformers, scipy, sklearn, matplotlib, seaborn, networkx; print('All OK')\"")
