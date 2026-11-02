"""统一入口：python -m src.run --help；绘图仅在显式 plot 命令中加载。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="光储容量规划与固定容量压力评估")
    commands = parser.add_subparsers(dest="command")
    baseline = commands.add_parser("baseline", help="求基准容量并保存逐时调度")
    baseline.add_argument("alpha", nargs="?", type=float, default=1.0)
    baseline.add_argument("gamma", nargs="?", type=float, default=cfg.BASELINE_GAMMA)
    baseline.add_argument("--out", type=Path, default=cfg.BASELINE_OUTPUT)
    baseline.add_argument("--threads", type=int, default=12)
    baseline.add_argument("--time-limit", type=float, default=1800.0)
    baseline.add_argument("--mutex-formulation", choices=("binary", "sos", "indicator"), default="sos")
    boundary = commands.add_parser("experiment1", aliases=["state_boundaries"], help="实验一：两条投资状态边界")
    for name, value in (("n_points", 20), ("workers", 1), ("alpha_probe_count", 7), ("gamma_probe_count", 5)):
        boundary.add_argument(name, nargs="?", type=int, default=value)
    boundary.add_argument("--preview", action="store_true")
    boundary.add_argument("--pilot", action="store_true")
    boundary.add_argument("--reset-checkpoint", action="store_true")
    boundary.add_argument("--out", type=Path, default=cfg.OUTPUT_ROOT, help="结果保存在此目录的 f2_e1 子目录")
    progress = commands.add_parser("boundary_progress", help="查看边界断点续算进度")
    progress.add_argument("--out", type=Path, default=cfg.OUTPUT_ROOT)
    lhs = commands.add_parser("experiment2", aliases=["stage2_lhs"], help="实验二：固定容量 LHS 与发生年份评估")
    lhs.add_argument("--scenarios", type=Path, help="可选已有 LHS 场景 CSV")
    lhs.add_argument("--baseline-summary", type=Path, default=cfg.BASELINE_OUTPUT / "run_summary.json")
    lhs.add_argument("--out", type=Path, default=cfg.LHS_OUTPUT)
    lhs.add_argument("--n", "--expected-n", dest="n", type=int, default=cfg.LHS_N)
    lhs.add_argument("--seed", type=int, default=cfg.LHS_SEED)
    lhs.add_argument("--threads", type=int, default=20)
    lhs.add_argument("--workers", type=int, default=1)
    lhs.add_argument("--time-limit", type=float, default=600.0)
    lhs.add_argument("--limit", type=int, help="仅运行部分场景，标记为试运行")
    lhs.add_argument("--gamma-upper", type=float, default=cfg.LHS_GAMMA_BOUNDS[1])
    lhs.add_argument("--mutex-formulation", choices=("binary", "sos", "indicator"), default="sos")
    lhs.add_argument("--prepare-only", action="store_true", help="只生成场景，不求解")
    sensitivity = commands.add_parser("sensitivity", help="实验一补充：19 个重优化敏感性情景")
    sensitivity.add_argument("--out", type=Path, default=cfg.SENSITIVITY_OUTPUT)
    sensitivity.add_argument("--threads", type=int, default=12)
    sensitivity.add_argument("--time-limit", type=float, default=1200.0)
    plot = commands.add_parser("plot", help="仅读取已有结果出图，不调用优化模型")
    plot.add_argument("--root", type=Path, default=cfg.OUTPUT_ROOT)
    plot.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "baseline":
        from src.experiment1 import run_baseline
        report = run_baseline(args.alpha, args.gamma, output=args.out, threads=args.threads,
                              time_limit=args.time_limit, mutex_formulation=args.mutex_formulation)
    elif args.command in ("experiment1", "state_boundaries"):
        from src.experiment1 import run_boundaries
        boundary_args = [str(args.n_points), str(args.workers), str(args.alpha_probe_count), str(args.gamma_probe_count)]
        boundary_args.extend(flag for flag, enabled in (
            ("--preview", args.preview), ("--pilot", args.pilot),
            ("--reset-checkpoint", args.reset_checkpoint)) if enabled)
        return run_boundaries(boundary_args, output_root=args.out)
    elif args.command == "boundary_progress":
        path = args.out / "f2_e1" / "e1_state_boundaries.gamma_tou.v5.checkpoint.progress.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_started", "path": str(path)}
    elif args.command in ("experiment2", "stage2_lhs"):
        from src.experiment2 import generate_scenarios, run
        if args.prepare_only:
            report = generate_scenarios(args.out / "scenarios", n=args.n, seed=args.seed,
                                        gamma_bounds=(0.4, args.gamma_upper))
        else:
            report = run(scenario_path=args.scenarios, baseline_summary=args.baseline_summary,
                         output=args.out, threads=args.threads, workers=args.workers,
                         mutex_formulation=args.mutex_formulation, time_limit=args.time_limit,
                         limit=args.limit, expected_n=args.n, seed=args.seed,
                         gamma_bounds=(0.4, args.gamma_upper))
    elif args.command == "sensitivity":
        from src.experiment1 import run_sensitivity
        report = run_sensitivity(args.out, threads=args.threads, time_limit=args.time_limit)
    else:
        from src.plots import all_figures
        report = all_figures(args.root, output=args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
