"""
Evaluation harness.

Metrics implemented from scratch (no external IR library) so the
computation is fully inspectable in an interview:

- Precision@K: of the top K ranked jobs, what fraction are relevant
  (labeled relevant in the eval set)?
- Recall@K: of all relevant jobs for this candidate, what fraction appear
  in the top K?
- NDCG@K: precision/recall treat relevance as binary and ignore rank
  position; NDCG rewards relevant results appearing *higher* in the list,
  which is what actually matters for "which job should I apply to first."

Relevance labels come from `data/eval_labels.json`: for a fixed set of
candidate/job pairs, a binary "would a reasonable candidate prioritize
applying to this job" label, hand-authored to approximate ground truth
(see README "Evaluation" for the labeling methodology and its limits).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class RankedResult:
    job_id: str
    score: float


def precision_at_k(ranked: list[RankedResult], relevant_ids: set[str], k: int) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r.job_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked: list[RankedResult], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k_ids = {r.job_id for r in ranked[:k]}
    hits = len(top_k_ids & relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(ranked: list[RankedResult], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(ranked[:k]):
        rel = 1.0 if r.job_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i=0 -> log2(2)=1
    return dcg


def ndcg_at_k(ranked: list[RankedResult], relevant_ids: set[str], k: int) -> float:
    dcg = dcg_at_k(ranked, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


class LatencyTimer:
    """Simple context-manager latency tracker in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def aggregate_metrics(
    per_query_ranked: list[list[RankedResult]],
    per_query_relevant: list[set[str]],
    ks: list[int],
) -> dict:
    result = {"precision_at_k": {}, "recall_at_k": {}, "ndcg_at_k": {}}
    for k in ks:
        precisions = [precision_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        recalls = [recall_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        result["precision_at_k"][k] = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
        result["recall_at_k"][k] = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        result["ndcg_at_k"][k] = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0
    return result
