"""Print the README results tables from a finished run.

    python tools/results_table.py                 # reads runs/result.json and runs/checks.json
    python tools/results_table.py runs_x/result.json

Only formats what harness/run_task.py and harness/check_rules.py wrote; it measures nothing.
"""

import json
import sys
from pathlib import Path


def main():
    result_path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = result["summary"]
    base, sol, ratios = summary.get("base", {}), summary.get("solution", {}), \
        summary.get("ratios", {})
    print("| | Base Soup | Ours | Ratio |")
    print("|---|---|---|---|")
    print(f"| `soup train` wall time, s (median of {sol.get('runs')} runs) | {base.get('wall_s')} "
          f"| {sol.get('wall_s')} | speedup x{ratios.get('speedup')} |")
    print(f"| 64 training steps, s | {base.get('train_loop_s')} | {sol.get('train_loop_s')} | "
          f"x{round(base['train_loop_s'] / sol['train_loop_s'], 3) if base.get('train_loop_s') and sol.get('train_loop_s') else '-'} |")
    print(f"| Peak GPU memory (nvidia-smi), MiB | {base.get('gpu_peak_mib')} | "
          f"{sol.get('gpu_peak_mib')} | base/ours x{ratios.get('memory_ratio')} |")
    print(f"| Eval loss (held-out rows) | {base.get('eval_loss')} | {sol.get('eval_loss')} | "
          f"{ratios.get('quality_delta_pct')} % vs base |")

    runs = result["runs"].get("solution", [])
    if runs:
        print("\n| Kernel | Share of steps 3+ (lowest run) |")
        print("|---|---|")
        names = sorted({n for r in runs for n in r["metrics"].get("own_kernel_ms", {})})
        for name in names:
            shares = [r["metrics"]["own_kernel_ms"].get(name, 0) / (10 * r["metrics"]["timed_loop_s"])
                      for r in runs]
            print(f"| `{name}` | {min(shares):.1f}% |")
        totals = [sum(r["metrics"]["own_kernel_ms"].values()) / (10 * r["metrics"]["timed_loop_s"])
                  for r in runs]
        print(f"| **all ours** | **{min(totals):.1f}%** |")

    checks_path = result_path.parent / "checks.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        print("\n| Check | Result |")
        print("|---|---|")
        for c in checks["checks"]:
            print(f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'}: {c['detail']} |")
    print(f"\nversions: {json.dumps(result.get('versions'))}")


if __name__ == "__main__":
    main()
