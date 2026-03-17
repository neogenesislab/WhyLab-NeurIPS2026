# -*- coding: utf-8 -*-
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

tests_passed = 0
tests_failed = 0

def ok(name):
    global tests_passed
    tests_passed += 1
    print("[PASS] " + name)

def fail(name, err):
    global tests_failed
    tests_failed += 1
    print("[FAIL] " + name + ": " + str(err))

# Test 1
try:
    from experiments.e1_drift_detection import CUSUM, PageHinkley
    import numpy as np
    c = CUSUM(h=5.0)
    rng = np.random.default_rng(42)
    vals = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    alarms = [c.update(v) for v in vals]
    assert sum(alarms) > 0
    ok("E1 CUSUM")
except Exception as e:
    fail("E1 CUSUM", e)

try:
    ph = PageHinkley(lambda_=50.0)
    alarms2 = [ph.update(v) for v in vals]
    ok("E1 PageHinkley")
except Exception as e:
    fail("E1 PageHinkley", e)

# Test 2
try:
    from experiments.e3a_stability import ctrl_adam
    state = {"g_sq_ema": None}
    g = np.array([1.0, 2.0, 3.0])
    for t in range(20):
        z = ctrl_adam(t=t, T=100, g_hat=g, state=state, lr=0.1)
    assert 0.01 <= z <= 0.8
    ok("E3a Adam")
except Exception as e:
    fail("E3a Adam", e)

# Test 3
try:
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("experiments/config.yaml").read_text(encoding="utf-8"))
    e5 = cfg["e5"]
    assert len(e5["ablations"]) == 7
    ok("Config E5")
except Exception as e:
    fail("Config E5", e)

# Test 4
try:
    from experiments.swebench_loader import SWEProblem, compute_patch_magnitude
    p = SWEProblem(instance_id="t", repo="a/b", base_commit="c",
                   problem_statement="d", hints_text="", test_patch="",
                   patch="", version="1", created_at="x")
    assert p.repo_name == "b"
    ok("SWE-bench loader")
except Exception as e:
    fail("SWE-bench loader", e)

# Test 5
try:
    from experiments.swebench_reflexion import _extract_patch
    r = _extract_patch("```diff\ndiff --git a/f b/f\n-old\n+new\n```")
    assert "diff" in r
    ok("SWE-bench reflexion")
except Exception as e:
    fail("SWE-bench reflexion", e)

# Test 6
try:
    from experiments.audit_layer import AgentAuditLayer
    a = AgentAuditLayer({"c1":True,"c2":True,"c3":True,
        "c1_window":5,"c1_agreement_threshold":0.4,
        "c2_e_thresh":1.5,"c2_rv_thresh":0.05,
        "c3_epsilon_floor":0.01,"c3_ceiling":0.8})
    d = a.evaluate_update(0.5, True, [0.3,0.4], [0.5], 0.1)
    assert d is not None
    ok("Audit layer")
except Exception as e:
    fail("Audit layer", e)

# Test 7
try:
    from experiments.e5_swebench_benchmark import load_config
    _, e5c = load_config()
    assert e5c["benchmark"] == "swe-bench-lite"
    ok("E5 benchmark")
except Exception as e:
    fail("E5 benchmark", e)

print("")
print("Results: " + str(tests_passed) + " passed, " + str(tests_failed) + " failed")
sys.exit(0 if tests_failed == 0 else 1)
