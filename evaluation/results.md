# Evaluation Results

Generated: 2026-08-27T18:59:58.061745+00:00Z
Source: `GET /api/v1/evaluation/run` against `http://localhost:8000`

| Mode | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | NDCG@1 | NDCG@3 | NDCG@5 | Avg latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keyword | 0.714 | 0.429 | 0.314 | 0.476 | 0.738 | 0.929 | 0.714 | 0.716 | 0.806 | 4.02 |
| Vector | 1.000 | 0.381 | 0.314 | 0.691 | 0.762 | 0.929 | 1.000 | 0.814 | 0.906 | 4.02 |
| Hybrid | 0.857 | 0.524 | 0.314 | 0.548 | 0.929 | 0.929 | 0.857 | 0.880 | 0.880 | 3.98 |

## Reading this table

- **Precision@K** — of the top K jobs shown, what fraction were labeled relevant.
- **Recall@K** — of all relevant jobs for that candidate, what fraction appeared in the top K.
- **NDCG@K** — like Precision/Recall but rewards relevant jobs appearing *higher* in the
  ranking, which is what "which job should I apply to first" actually depends on.
- **Avg latency (ms)** — average end-to-end ranking latency per candidate query
  (retrieval + scoring + RAG explanation generation), measured server-side.

See `backend/data/eval_labels.json` for the labeled test set and its labeling methodology.
