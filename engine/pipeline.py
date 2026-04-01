# -*- coding: utf-8 -*-
"""WhyLab 파이프라인 진입점.

커맨드라인에서 실행 가능한 스크립트입니다.
사용법:
    python -m engine --scenario A
    python -m engine --scenario A --debate
    python -m engine --benchmark ihdp acic jobs
    python -m engine --benchmark ihdp --output results/ --latex
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# EconML/sklearn 반복 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="econml")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from engine.orchestrator import Orchestrator


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _save_results(data: dict, output_dir: str, prefix: str) -> None:
    """결과를 JSON으로 저장."""
    import numpy as np

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"직렬화 불가: {type(obj)}")

    filepath = out_path / f"{prefix}_results.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_convert, indent=2, ensure_ascii=False)

    logging.getLogger("whylab.cli").info("💾 결과 저장: %s", filepath)


def _generate_latex_table(results: dict, output_dir: str) -> None:
    """논문용 LaTeX 비교표 자동 생성."""
    out_path = Path(output_dir) / "tables"
    out_path.mkdir(parents=True, exist_ok=True)

    ds_names = list(results.keys())
    all_methods = set()
    for ds_result in results.values():
        all_methods.update(ds_result.keys())

    ordered = ["S-Learner", "T-Learner", "X-Learner", "DR-Learner",
               "R-Learner", "LinearDML", "Ensemble"]
    methods = [m for m in ordered if m in all_methods]

    col_format = "l" + "cc" * len(ds_names)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{WhyLab Meta-Learner Benchmark Results}",
        r"\label{tab:benchmark}",
        f"\\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
    ]

    h1 = "Method"
    h2 = ""
    for ds in ds_names:
        h1 += f" & \\multicolumn{{2}}{{c}}{{{ds.upper()}}}"
        h2 += r" & $\sqrt{\text{PEHE}}$ & ATE Bias"
    lines.append(h1 + r" \\")
    lines.append(r"\cmidrule(lr){2-" + str(len(ds_names) * 2 + 1) + "}")
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for method in methods:
        row = method
        for ds in ds_names:
            r = results.get(ds, {}).get(method)
            if r:
                row += f" & ${r['pehe_mean']:.3f} \\pm {r['pehe_std']:.3f}$"
                row += f" & ${r['ate_bias_mean']:.3f} \\pm {r['ate_bias_std']:.3f}$"
            else:
                row += " & --- & ---"
        lines.append(row + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    filepath = out_path / "benchmark_table.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.getLogger("whylab.cli").info("📄 LaTeX 테이블: %s", filepath)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="whylab",
        description="WhyLab - Causal Inference Research Pipeline",
    )

    # 모드 선택
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--scenario", "-s",
        choices=["A", "B"],
        default="A",
        help="파이프라인 시나리오 (A: 신용한도, B: 마케팅)",
    )
    group.add_argument(
        "--benchmark", "-b",
        nargs="+",
        choices=["ihdp", "acic", "jobs"],
        help="학술 벤치마크 실행",
    )

    parser.add_argument("--debate", "-d", action="store_true",
                        help="Multi-Agent Debate 활성화")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="결과 저장 디렉토리")
    parser.add_argument("--replications", "-r", type=int, default=10,
                        help="벤치마크 반복 횟수 (기본: 10)")
    parser.add_argument("--latex", action="store_true",
                        help="LaTeX 비교표 자동 생성")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="디버그 로깅")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(verbose=True)

    if args.benchmark:
        # 벤치마크 모드
        from engine.config import WhyLabConfig
        from engine.cells.benchmark_cell import BenchmarkCell

        config = WhyLabConfig()
        config.benchmark.datasets = args.benchmark
        config.benchmark.n_replications = args.replications

        cell = BenchmarkCell(config)
        result = cell.execute({})

        # 결과 저장 (print 전에 수행하여 인코딩 오류 시에도 파일 보존)
        if args.output:
            _save_results(result["benchmark_results"], args.output, "benchmark")
            if args.latex:
                _generate_latex_table(result["benchmark_results"], args.output)

        print("\n" + "=" * 60)
        print("[BENCHMARK] WhyLab Benchmark Results")
        print("=" * 60)
        print(result["benchmark_table"])
    else:
        # 파이프라인 모드
        orchestrator = Orchestrator()
        try:
            result = orchestrator.run_pipeline(scenario=args.scenario)
            if args.output:
                _save_results(result, args.output, f"pipeline_{args.scenario}")
        except Exception:
            sys.exit(1)


if __name__ == "__main__":
    main()

