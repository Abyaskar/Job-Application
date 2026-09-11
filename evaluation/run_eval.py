"""
Standalone evaluation runner.

Usage:
    python evaluation/run_eval.py --base-url http://localhost:8000

Hits the running API's /api/v1/evaluation/run endpoint (so it exercises the
exact same code path the product uses, not a separate offline copy of the
ranking logic), and writes a formatted comparison report to
evaluation/results.md. This is what the README's "Evaluation Results"
numbers are generated from -- re-run this after any ranking change to keep
the README honest.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def render_markdown_table(results: dict) -> str:
    modes = ["keyword", "vector", "hybrid"]
    ks = sorted({int(k) for m in results.values() for k in m["precision_at_k"].keys()})

    lines = ["| Mode | " + " | ".join(f"P@{k}" for k in ks) + " | " + " | ".join(f"R@{k}" for k in ks)
             + " | " + " | ".join(f"NDCG@{k}" for k in ks) + " | Avg latency (ms) |"]
    lines.append("|---" * (1 + 3 * len(ks) + 1) + "|")

    for mode in modes:
        if mode not in results:
            continue
        m = results[mode]
        row = [mode.capitalize()]
        row += [f"{m['precision_at_k'][str(k)]:.3f}" for k in ks]
        row += [f"{m['recall_at_k'][str(k)]:.3f}" for k in ks]
        row += [f"{m['ndcg_at_k'][str(k)]:.3f}" for k in ks]
        row.append(f"{m['avg_latency_ms']:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evaluation/results.md")
    args = parser.parse_args()

    url = f"{args.base_url}/api/v1/evaluation/run?top_k={args.top_k}"
    print(f"Fetching evaluation results from {url} ...")
    try:
        results = fetch_json(url)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to reach API: {exc}", file=sys.stderr)
        print("Is the backend running? Try: uvicorn app.main:app --reload (from backend/)", file=sys.stderr)
        sys.exit(1)

    if "error" in results:
        print(f"API returned an error: {results['error']}", file=sys.stderr)
        sys.exit(1)

    table = render_markdown_table(results)
    report = f"""# Evaluation Results

Generated: {datetime.now(timezone.utc).isoformat()}Z
Source: `GET /api/v1/evaluation/run` against `{args.base_url}`

{table}

## Reading this table

- **Precision@K** — of the top K jobs shown, what fraction were labeled relevant.
- **Recall@K** — of all relevant jobs for that candidate, what fraction appeared in the top K.
- **NDCG@K** — like Precision/Recall but rewards relevant jobs appearing *higher* in the
  ranking, which is what "which job should I apply to first" actually depends on.
- **Avg latency (ms)** — average end-to-end ranking latency per candidate query
  (retrieval + scoring + RAG explanation generation), measured server-side.

See `backend/data/eval_labels.json` for the labeled test set and its labeling methodology.
"""
    with open(args.out, "w") as f:
        f.write(report)

    print(report)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
