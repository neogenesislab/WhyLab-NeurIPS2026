"""
WhyLab Zenodo Upload Script
- Create new deposition → upload PDF → set metadata → publish
- Reuses ZENODO_ACCESS_TOKEN from EthicaAI .env

Usage:
    python scripts/zenodo_upload.py              # 업로드 (퍼블리시 안 함)
    python scripts/zenodo_upload.py --publish     # 업로드 + 퍼블리시
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

# ===== Constants =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "paper" / "main.pdf"
PDF_FILENAME = "WhyLab - Causal Audit Framework for Stable Agent Self-Improvement.pdf"

ZENODO_API_BASE = "https://zenodo.org/api"

# Try WhyLab .env first, fall back to EthicaAI .env
ENV_PATHS = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT.parent / "EthicaAI" / ".env",
    Path(r"d:\00.test\PAPER\EthicaAI\.env"),
]


def load_config():
    """Load Zenodo token from available .env files."""
    for env_path in ENV_PATHS:
        if env_path.exists():
            load_dotenv(env_path)
            break

    token = os.getenv("ZENODO_ACCESS_TOKEN")
    if not token:
        print("❌ ZENODO_ACCESS_TOKEN not found")
        sys.exit(1)
    return token


def create_deposition(token):
    """Create a new empty deposition on Zenodo."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{ZENODO_API_BASE}/deposit/depositions",
        data=json.dumps({}),
        headers=headers,
    )
    resp.raise_for_status()
    dep = resp.json()
    print(f"   ✅ Deposition 생성: #{dep['id']}")
    return dep


def upload_pdf(token, draft, pdf_path):
    """Upload PDF to deposition."""
    headers = {"Authorization": f"Bearer {token}"}
    bucket_url = draft["links"]["bucket"]

    print(f"\n📤 PDF 업로드 중... ({pdf_path.name})")
    with open(pdf_path, "rb") as fp:
        resp = requests.put(
            f"{bucket_url}/{PDF_FILENAME}",
            data=fp,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
        )
    resp.raise_for_status()
    result = resp.json()
    size_kb = result["size"] / 1024
    print(f"   ✅ 업로드 완료: {PDF_FILENAME} ({size_kb:.0f} KB)")
    return result


def update_metadata(token, draft):
    """Set deposition metadata."""
    deposition_id = draft["id"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    metadata = {
        "metadata": {
            "title": "WhyLab: A Causal Audit Framework for Stable Agent Self-Improvement",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": (
                "<p>Self-improving AI agents risk cognitive policy oscillation "
                "when noisy feedback causes strategy degradation. "
                "<strong>WhyLab</strong> provides a causal safety monitoring framework "
                "with three components:</p>"
                "<ul>"
                "<li><strong>C1</strong>: Information-theoretic drift detection</li>"
                "<li><strong>C2</strong>: E-value sensitivity filter for fragile successes</li>"
                "<li><strong>C3</strong>: Lyapunov-bounded adaptive damping</li>"
                "</ul>"
                "<p>On SWE-bench Lite (300 problems), C2 maintains zero regressions "
                "across 10,500 episodes. A non-stationary experiment (E6) validates "
                "all three components independently: C3 reduces energy by 49%, "
                "C2 reduces oscillation by 76%, C1 cuts drift detection delay by 16%.</p>"
                "<p>Code: https://github.com/Yesol-Pilot/WhyLab</p>"
            ),
            "creators": [{"name": "Anonymous", "affiliation": "Anonymous"}],
            "keywords": [
                "Causal Inference",
                "Agent Self-Improvement",
                "Drift Detection",
                "Sensitivity Analysis",
                "Lyapunov Stability",
                "E-value",
                "NeurIPS 2026",
            ],
            "access_right": "open",
            "license": "MIT",
            "publication_date": "2026-03-17",
            "version": "2.0.0",
            "language": "eng",
            "notes": (
                "Anonymous preprint for NeurIPS 2026. "
                "17 pages (9 content + refs + appendix + checklist). "
                "6 experiments (E1-E6), 840+ episodes with bootstrap CI."
            ),
        }
    }

    print("\n📋 메타데이터 업데이트 중...")
    resp = requests.put(
        f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}",
        data=json.dumps(metadata),
        headers=headers,
    )
    resp.raise_for_status()
    print("   ✅ 메타데이터 업데이트 완료")
    return resp.json()


def publish(token, draft):
    """Publish the deposition."""
    deposition_id = draft["id"]
    headers = {"Authorization": f"Bearer {token}"}

    print("\n🚀 퍼블리시 중...")
    resp = requests.post(
        f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers,
    )
    resp.raise_for_status()
    result = resp.json()
    doi = result.get("doi", "N/A")
    print(f"   ✅ 퍼블리시 완료!")
    print(f"   DOI: {doi}")
    print(f"   URL: https://zenodo.org/records/{result['id']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="WhyLab Zenodo Upload")
    parser.add_argument("--publish", action="store_true", help="Publish after upload")
    args = parser.parse_args()

    token = load_config()

    print("=" * 60)
    print("WhyLab Zenodo Upload")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"❌ PDF not found: {PDF_PATH}")
        sys.exit(1)
    print(f"📄 PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")

    # Step 1: Create deposition
    print("\n🔍 Zenodo deposition 생성 중...")
    draft = create_deposition(token)

    # Step 2: Upload PDF
    upload_pdf(token, draft, PDF_PATH)

    # Step 3: Update metadata
    update_metadata(token, draft)

    # Step 4: Publish (optional)
    if args.publish:
        publish(token, draft)
    else:
        print(f"\n⚠️  퍼블리시하지 않았습니다. 확인 후:")
        print(f"   python scripts/zenodo_upload.py --publish")
        print(f"   또는 Zenodo 웹에서 직접 Publish")

    print("\n" + "=" * 60)
    print("완료!")


if __name__ == "__main__":
    main()
