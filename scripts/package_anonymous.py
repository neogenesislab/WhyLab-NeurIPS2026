import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_NAME = "WhyLab_NeurIPS2026_anonymous.zip"

INCLUDE_DIRS = ["paper", "experiments"]
INCLUDE_FILES = ["requirements.txt", "README_ANON.md", "example_usage.py"]
EXCLUDE_DIRS = [".git", ".github", "paper/figs_raw", "paper/.aux", "dashboard", "engine"]
EXCLUDE_EXTS = [".aux", ".bbl", ".blg", ".log", ".out", ".fdb_latexmk", ".fls", ".synctex.gz", ".toc", ".pyc", ".db"]

def is_excluded(path):
    path_str = str(path.relative_to(ROOT)).replace("\\", "/")
    
    for ex in EXCLUDE_DIRS:
        if path_str.startswith(ex) or f"/{ex}/" in f"/{path_str}":
            return True
            
    if path.suffix in EXCLUDE_EXTS:
        return True
        
    if "whylab.db" in path.name or ".env" in path.name:
        return True
        
    return False

def main():
    out_path = ROOT / ARCHIVE_NAME
    if out_path.exists():
        out_path.unlink()
        
    print(f"📦 Creating anonymous review package: {ARCHIVE_NAME}...")
    
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add included files
        for fname in INCLUDE_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                arcname = "WhyLab_Anonymous/" + fname
                zf.write(fpath, arcname)
                print(f"  + {arcname}")
                
        # Add included directories
        for dname in INCLUDE_DIRS:
            dpath = ROOT / dname
            if not dpath.exists():
                continue
                
            for root, _, files in os.walk(dpath):
                root_path = Path(root)
                if is_excluded(root_path):
                    continue
                    
                for file in files:
                    file_path = root_path / file
                    if is_excluded(file_path):
                        continue
                        
                    arcname = "WhyLab_Anonymous/" + str(file_path.relative_to(ROOT)).replace("\\", "/")
                    zf.write(file_path, arcname)
                    print(f"  + {arcname}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Packaged successfully: {size_mb:.2f} MB")
    print(f"➡️  Upload via OpenReview")

if __name__ == "__main__":
    main()
