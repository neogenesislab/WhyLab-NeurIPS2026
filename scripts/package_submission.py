"""
WhyLab Submission Packaging Script
====================================
Creates three submission packages:

1. Personal Git   — commit + push (already done, push manually if 403)
2. Anonymous Git  — stripped of identity for blind review
3. Zenodo archive — for persistent DOI

Usage:
    python scripts/package_submission.py

Output:
    submission/                     ← submission root
    submission/anonymous/           ← anonymized repo copy
    submission/whylab-neurips2026.zip  ← Zenodo-ready archive
    submission/supplemental.zip     ← NeurIPS supplemental material
"""
import shutil
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"

# -------------------------------------------------------------------
# Files to include in submission (whitelist approach for safety)
# -------------------------------------------------------------------
INCLUDE_DIRS = [
    "engine",
    "experiments",
    "scripts",
    "tests",
    "docs",
    "paper",
]

INCLUDE_ROOT_FILES = [
    "PROJECT_SPEC.md",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    ".gitignore",
]

# Files/patterns to EXCLUDE (security + cleanliness)
EXCLUDE_PATTERNS = [
    ".env",
    ".env.*",
    "__pycache__",
    "*.pyc",
    ".git",
    "*.aux",
    "*.log",
    "*.bbl",
    "*.blg",
    "*.synctex*",
    "*.fdb_latexmk",
    "*.fls",
    "submission",
    "tmp_test",
    "stats",
    "node_modules",
    ".vscode",
    ".idea",
]

# Identity strings to scrub for anonymous version
IDENTITY_STRINGS = [
    ("Yesol-Pilot", "Anonymous"),
    ("yesol", "anonymous"),
    ("heoyesol", "anonymous"),
    ("Heo Yesol", "Anonymous Author"),
    ("허예솔", "Anonymous Author"),
]


def should_exclude(path: Path) -> bool:
    """Check if path matches any exclusion pattern."""
    name = path.name
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif name == pat:
            return True
    return False


def copy_tree_filtered(src: Path, dst: Path):
    """Copy directory tree with exclusion filtering."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if should_exclude(item):
            continue
        target = dst / item.name
        if item.is_dir():
            copy_tree_filtered(item, target)
        else:
            shutil.copy2(item, target)


def anonymize_file(filepath: Path):
    """Scrub identity strings from a file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return  # Skip binary files

    original = content
    for old, new in IDENTITY_STRINGS:
        content = content.replace(old, new)

    # Also remove git remote URLs that reveal identity
    content = re.sub(
        r"https://github\.com/[A-Za-z0-9_-]+/WhyLab",
        "https://github.com/anonymous/whylab-submission",
        content,
    )

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        print(f"  [anonymized] {filepath.relative_to(SUBMISSION)}")


def create_anonymous_copy():
    """Create anonymized copy of the repository."""
    print("\n" + "=" * 60)
    print("  Step 1: Creating anonymous copy")
    print("=" * 60)

    anon_dir = SUBMISSION / "anonymous"
    if anon_dir.exists():
        shutil.rmtree(anon_dir)
    anon_dir.mkdir(parents=True)

    # Copy whitelisted directories
    for dirname in INCLUDE_DIRS:
        src = ROOT / dirname
        if src.exists():
            copy_tree_filtered(src, anon_dir / dirname)
            print(f"  ✓ {dirname}/")

    # Copy whitelisted root files
    for fname in INCLUDE_ROOT_FILES:
        src = ROOT / fname
        if src.exists():
            shutil.copy2(src, anon_dir / fname)
            print(f"  ✓ {fname}")

    # Anonymize all text files
    print("\n  Anonymizing...")
    for fpath in anon_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix in (
            ".py", ".md", ".tex", ".bib", ".yaml", ".yml",
            ".txt", ".cfg", ".toml", ".json",
        ):
            anonymize_file(fpath)

    # Verify: main.tex should have "Anonymous Author(s)"
    main_tex = anon_dir / "paper" / "main.tex"
    if main_tex.exists():
        content = main_tex.read_text(encoding="utf-8")
        if "Anonymous Author" in content:
            print("  ✓ main.tex: anonymous authorship confirmed")
        else:
            print("  ⚠ main.tex: check author field!")

    return anon_dir


def create_zenodo_archive(anon_dir: Path):
    """Create Zenodo-ready zip archive from anonymous copy."""
    print("\n" + "=" * 60)
    print("  Step 2: Creating Zenodo archive")
    print("=" * 60)

    archive_name = "whylab-neurips2026"
    archive_path = SUBMISSION / archive_name

    shutil.make_archive(
        str(archive_path),
        "zip",
        root_dir=str(anon_dir.parent),
        base_dir=anon_dir.name,
    )

    zip_path = SUBMISSION / f"{archive_name}.zip"
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {zip_path.name} ({size_mb:.1f} MB)")
    return zip_path


def create_supplemental(anon_dir: Path):
    """Create NeurIPS supplemental material zip."""
    print("\n" + "=" * 60)
    print("  Step 3: Creating supplemental material")
    print("=" * 60)

    supp_dir = SUBMISSION / "supplemental_staging"
    if supp_dir.exists():
        shutil.rmtree(supp_dir)
    supp_dir.mkdir()

    # Include key files for reviewers
    supp_items = [
        ("scripts/reproduce_revision.py", "reproduce_revision.py"),
        ("experiments/e2_refutation.py", "experiments/e2_refutation.py"),
        ("experiments/e3a_ablation.py", "experiments/e3a_ablation.py"),
        ("experiments/invariance_check.py", "experiments/invariance_check.py"),
        ("experiments/aggregate_stats.py", "experiments/aggregate_stats.py"),
        ("experiments/config.yaml", "experiments/config.yaml"),
        ("docs/causal_assumptions.md", "docs/causal_assumptions.md"),
    ]

    # Copy results
    results_src = anon_dir / "experiments" / "results"
    results_dst = supp_dir / "experiments" / "results"
    if results_src.exists():
        shutil.copytree(results_src, results_dst)
        print(f"  ✓ experiments/results/")

    for src_rel, dst_rel in supp_items:
        src = anon_dir / src_rel
        if src.exists():
            dst = supp_dir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ✓ {dst_rel}")

    # Create a brief README
    readme = supp_dir / "README.md"
    readme.write_text(
        "# WhyLab — Supplemental Material\n\n"
        "## Reproduction\n\n"
        "```bash\n"
        "pip install numpy pandas scipy pyyaml\n"
        "python reproduce_revision.py\n"
        "```\n\n"
        "This runs all revision experiments (P0: refutation + ablation,\n"
        "P1: statistics aggregation + invariance check) and generates\n"
        "LaTeX tables in `paper/tables/`.\n\n"
        "Expected runtime: ~2 minutes on a standard laptop.\n\n"
        "## File Overview\n\n"
        "| File | Description |\n"
        "|:-----|:------------|\n"
        "| `reproduce_revision.py` | One-click reproduction script |\n"
        "| `experiments/e2_refutation.py` | DoWhy-aligned refutation tests |\n"
        "| `experiments/e3a_ablation.py` | Controller component ablation |\n"
        "| `experiments/invariance_check.py` | Conclusion invariance check |\n"
        "| `experiments/config.yaml` | Experiment configuration |\n"
        "| `docs/causal_assumptions.md` | Full assumption table |\n"
        "| `experiments/results/` | Pre-computed result CSVs |\n",
        encoding="utf-8",
    )
    print(f"  ✓ README.md")

    # Create zip
    archive_path = SUBMISSION / "supplemental"
    shutil.make_archive(str(archive_path), "zip", str(supp_dir))

    zip_path = SUBMISSION / "supplemental.zip"
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ supplemental.zip ({size_mb:.1f} MB)")

    # Cleanup staging
    shutil.rmtree(supp_dir)
    return zip_path


def print_instructions(zenodo_zip: Path, supp_zip: Path, anon_dir: Path):
    """Print manual steps for the user."""
    print("\n" + "=" * 60)
    print("  NEXT STEPS (manual)")
    print("=" * 60)

    print("""
  ┌─────────────────────────────────────────────────────┐
  │  1. PERSONAL GIT (push)                             │
  │     git push neurips main                           │
  │     (if 403: check GitHub PAT / SSH key)            │
  ├─────────────────────────────────────────────────────┤
  │  2. ANONYMOUS GIT (for blind review)                │
  │     a) Create new GitHub repo (e.g. whylab-anon)    │
  │     b) cd submission/anonymous                      │
  │        git init; git add -A                         │
  │        git commit -m "NeurIPS 2026 submission"      │
  │        git remote add origin <anon-repo-url>        │
  │        git push -u origin main                      │
  ├─────────────────────────────────────────────────────┤
  │  3. ZENODO (DOI)                                    │
  │     a) Go to https://zenodo.org/deposit/new         │
  │     b) Upload: whylab-neurips2026.zip               │
  │     c) Title: "WhyLab: Causal Auditing for Stable   │
  │        Self-Improving Agents (Code & Data)"         │
  │     d) Authors: Anonymous (for blind review period)  │
  │     e) License: MIT or Apache-2.0                   │
  │     f) Publish → get DOI                            │
  │     g) Add DOI to paper via footnote or checklist   │
  ├─────────────────────────────────────────────────────┤
  │  4. NeurIPS SUBMISSION                              │
  │     a) Upload: paper/main.pdf (논문+부록+체크리스트) │
  │     b) Upload: supplemental.zip                     │
  │     c) (Optional) Add Zenodo DOI or anonymous repo  │
  │        link in the paper/checklist                  │
  └─────────────────────────────────────────────────────┘
""")


def main():
    print("=" * 60)
    print("  WhyLab Submission Packaging")
    print("=" * 60)

    if SUBMISSION.exists():
        shutil.rmtree(SUBMISSION)
    SUBMISSION.mkdir()

    anon_dir = create_anonymous_copy()
    zenodo_zip = create_zenodo_archive(anon_dir)
    supp_zip = create_supplemental(anon_dir)
    print_instructions(zenodo_zip, supp_zip, anon_dir)

    print(f"\n  All outputs in: {SUBMISSION}")
    print("  Done! ✅")


if __name__ == "__main__":
    main()
