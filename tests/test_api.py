"""Sprint 29-34 API Integration Test."""
import requests
import json

base = "http://localhost:4001"
results = []

# 1. DB Status
try:
    r = requests.get(f"{base}/system/db/status", timeout=5)
    detail = r.json().get("hot", {}).get("record_count", "?") if r.ok else "FAIL"
    results.append(("DB Status", r.status_code, detail))
except Exception as e:
    results.append(("DB Status", 0, str(e)))

# 2. STEAM DGPs
try:
    r = requests.get(f"{base}/system/steam/dgps", timeout=5)
    detail = len(r.json().get("available_dgps", [])) if r.ok else "FAIL"
    results.append(("STEAM DGPs", r.status_code, f"{detail} templates"))
except Exception as e:
    results.append(("STEAM DGPs", 0, str(e)))

# 3. Agent Registry
try:
    r = requests.get(f"{base}/system/agent-registry", timeout=5)
    detail = sum(len(v) for v in r.json().values()) if r.ok else "FAIL"
    results.append(("Agent Registry", r.status_code, f"{detail} agents"))
except Exception as e:
    results.append(("Agent Registry", 0, str(e)))

# 4. Architect Diagnose
try:
    r = requests.get(f"{base}/system/architect/diagnose", timeout=5)
    detail = r.json().get("health_score", "?") if r.ok else "FAIL"
    results.append(("Architect", r.status_code, f"Score={detail}/100"))
except Exception as e:
    results.append(("Architect", 0, str(e)))

# 5. SaaS Readiness
try:
    r = requests.get(f"{base}/system/saas/readiness", timeout=5)
    detail = r.json().get("readiness_score", "?") if r.ok else "FAIL"
    results.append(("SaaS Ready", r.status_code, f"{detail}%"))
except Exception as e:
    results.append(("SaaS Ready", 0, str(e)))

# 6. Constitution Info
try:
    r = requests.get(f"{base}/system/constitution/info", timeout=5)
    detail = r.json().get("version", "?") if r.ok else "FAIL"
    results.append(("Constitution", r.status_code, detail))
except Exception as e:
    results.append(("Constitution", 0, str(e)))

# 7. STEAM Generate
try:
    r = requests.post(f"{base}/system/steam/generate?dgp_name=labor_market&n=1000&seed=42", timeout=10)
    if r.ok:
        d = r.json()
        ate = d.get("ate_true", 0)
        grade = d.get("quality_metrics", {}).get("quality_grade", "?")
        results.append(("STEAM Gen", r.status_code, f"ATE={ate:.3f} Grade={grade}"))
    else:
        results.append(("STEAM Gen", r.status_code, "FAIL"))
except Exception as e:
    results.append(("STEAM Gen", 0, str(e)))

# 8. Paper Draft
try:
    r = requests.post(f"{base}/system/paper/draft?grand_challenge_id=GC_002", timeout=10)
    detail = r.json().get("title", "?")[:60] if r.ok else "FAIL"
    results.append(("Paper Draft", r.status_code, detail))
except Exception as e:
    results.append(("Paper Draft", 0, str(e)))

# 9. Migration Plan
try:
    r = requests.get(f"{base}/system/saas/migration-plan", timeout=5)
    detail = r.json().get("total_items", "?") if r.ok else "FAIL"
    results.append(("Migration", r.status_code, f"{detail} items"))
except Exception as e:
    results.append(("Migration", 0, str(e)))

print("=== API Integration Test Results ===")
for name, status, detail in results:
    mark = "PASS" if status == 200 else "FAIL"
    print(f"[{mark}] {name}: {status} | {detail}")

passed = sum(1 for _, s, _ in results if s == 200)
print(f"\nTotal: {passed}/{len(results)} passed")
