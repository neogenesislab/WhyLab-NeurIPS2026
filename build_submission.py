# -*- coding: utf-8 -*-
"""Build NeurIPS 2026 submission ZIP package."""
import zipfile
import os
from pathlib import Path

ROOT = Path(r"d:\00.test\PAPER\WhyLab")
OUT_DIR = ROOT / "submission"
OUT_DIR.mkdir(exist_ok=True)

# === 1. Main paper PDF ===
paper_zip = OUT_DIR / "whylab_neurips2026_paper.zip"
with zipfile.ZipFile(paper_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    # Main PDF
    zf.write(ROOT / "paper" / "main.pdf", "main.pdf")
    # LaTeX source (for camera-ready)
    zf.write(ROOT / "paper" / "main.tex", "latex/main.tex")
    zf.write(ROOT / "paper" / "references.bib", "latex/references.bib")
    # NeurIPS style file
    sty = ROOT / "paper" / "neurips_2025.sty"
    if sty.exists():
        zf.write(sty, "latex/neurips_2025.sty")

print(f"[1] Paper ZIP: {paper_zip} ({paper_zip.stat().st_size // 1024} KB)")

# === 2. Supplementary materials (code + results) ===
supp_zip = OUT_DIR / "whylab_neurips2026_supplementary.zip"

# Files/dirs to include from experiments/
INCLUDE_EXTS = {".py", ".yaml", ".txt", ".md"}
EXCLUDE_DIRS = {"__pycache__", "cache", "data", ".git"}

with zipfile.ZipFile(supp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    # README for supplementary
    readme = """# WhyLab: Supplementary Code

## Structure
- `experiments/` - All experiment scripts (E1-E5)
- `experiments/results/` - Pre-computed results (CSV)
- `experiments/prompts/` - LLM prompts for E4/E5
- `experiments/config.yaml` - Experiment configuration

## Reproduction
```bash
pip install numpy pandas scipy pyyaml datasets google-generativeai
# Set GEMINI_API_KEY in .env or environment
python -m experiments.e1_drift_detection
python -m experiments.e3a_stability
python -m experiments.e5_swebench_benchmark --pilot
```
"""
    zf.writestr("README.md", readme)

    # Experiment code
    exp_dir = ROOT / "experiments"
    for f in sorted(exp_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(ROOT)
            # Skip excluded dirs
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            # Include only relevant extensions + CSV results
            if f.suffix in INCLUDE_EXTS or f.suffix == ".csv":
                zf.write(f, str(rel))

    # Include .env.example (not .env!)
    env_example = "# API Key (required for E4/E5 experiments)\nGEMINI_API_KEY=your-api-key-here\n"
    zf.writestr(".env.example", env_example)

print(f"[2] Supplementary ZIP: {supp_zip} ({supp_zip.stat().st_size // 1024} KB)")

# === 3. List contents ===
print("\n--- Paper ZIP contents ---")
with zipfile.ZipFile(paper_zip, "r") as zf:
    for info in zf.infolist():
        print(f"  {info.filename} ({info.file_size // 1024} KB)")

print("\n--- Supplementary ZIP contents ---")
with zipfile.ZipFile(supp_zip, "r") as zf:
    for info in zf.infolist():
        if info.file_size > 0:
            print(f"  {info.filename} ({info.file_size // 1024} KB)")

print(f"\nTotal package: {(paper_zip.stat().st_size + supp_zip.stat().st_size) // 1024} KB")
print("Ready for OpenReview submission!")
