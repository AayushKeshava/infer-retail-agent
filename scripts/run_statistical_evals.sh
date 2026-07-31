#!/usr/bin/env bash

N=10
mkdir -p reports/statistical_runs

rm -f \
  reports/statistical_runs/baseline_*.json \
  reports/statistical_runs/improved_*.json

for version in baseline improved; do
  for run in $(seq 1 "$N"); do
    echo "Running $version: $run/$N"

    uv run python scripts/create_eval_db.py >/dev/null

    PROMPT_VERSION="$version" uv run pytest \
      tests/test_behavior_evals.py -q \
      --json-report \
      --json-report-file="reports/statistical_runs/${version}_${run}.json" \
      || true
  done
done

uv run python - <<'PY'
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(passed: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    z = 1.96
    p = passed / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - margin, centre + margin


rows = []

for version in ("baseline", "improved"):
    results = defaultdict(lambda: {"passed": 0, "total": 0})

    for filename in glob.glob(
        f"reports/statistical_runs/{version}_*.json"
    ):
        report = json.loads(Path(filename).read_text())

        for test in report.get("tests", []):
            outcome = test.get("outcome")

            if outcome not in {"passed", "failed"}:
                continue

            test_name = test["nodeid"].split("::")[-1]
            results[test_name]["total"] += 1

            if outcome == "passed":
                results[test_name]["passed"] += 1

    for test_name, counts in sorted(results.items()):
        passed = counts["passed"]
        total = counts["total"]
        rate = passed / total
        low, high = wilson_interval(passed, total)

        rows.append(
            {
                "prompt": version,
                "test": test_name,
                "passed": passed,
                "runs": total,
                "pass_rate": f"{rate:.1%}",
                "ci_95": f"{low:.1%}–{high:.1%}",
            }
        )

output = Path("reports/statistical_summary.csv")

with output.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"{'Prompt':<10} {'Test':<58} {'Result':<10} {'95% CI'}")
print("-" * 100)

for row in rows:
    print(
        f"{row['prompt']:<10} "
        f"{row['test']:<58} "
        f"{row['passed']}/{row['runs']:<7} "
        f"{row['ci_95']}"
    )

print(f"\nSaved to {output}")
PY