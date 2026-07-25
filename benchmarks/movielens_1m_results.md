# MovieLens-1M implicit-ranking benchmark

Run date: 2026-07-24

This report compares Random, Popularity, BPR, NeuMF, Item-KNN, and ALS using the
same full-catalog ranking protocol on the
[MovieLens-1M dataset](https://grouplens.org/datasets/movielens/1m/).
Hyperparameters were fixed before evaluation and were not tuned on the test set.

## Summary

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MAP@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.00549 | 0.00254 | 0.05220 | 0.00182 | 0.01644 | 0.00580 |
| Popularity | 0.07019 | 0.04265 | 0.36808 | 0.03665 | 0.16305 | 0.07914 |
| BPR | 0.06042 | 0.05994 | 0.41531 | 0.03231 | 0.15642 | 0.07528 |
| NeuMF | 0.07060 | 0.06036 | 0.45293 | 0.03512 | 0.17168 | 0.08309 |
| **Item-KNN** | **0.08626** | 0.06355 | 0.46221 | **0.04768** | **0.20279** | **0.10134** |
| **ALS** | 0.08280 | **0.07599** | **0.50232** | 0.04432 | 0.20255 | 0.10015 |

Item-KNN produced the best Precision, MAP, MRR, and NDCG at 10. ALS produced
the best Recall and Hit Rate. BPR and NeuMF both exceeded Popularity on Recall
and Hit Rate but remained behind Item-KNN and ALS. The requested higher
iteration counts made both models substantially slower. In particular, NeuMF
at 100 epochs scored below its earlier 15-epoch run on every metric, consistent
with overfitting under this fixed configuration.

## Runtime

| Model | Fit time (s) | Ranking/evaluation time (s) | Total model time (s) |
|---|---:|---:|---:|
| Random | <0.001 | 0.616 | 0.616 |
| Popularity | 0.001 | 0.418 | 0.418 |
| BPR | 94.689 | 0.626 | 95.315 |
| NeuMF | 1806.349 | 91.983 | 1898.332 |
| Item-KNN | 0.054 | 0.781 | 0.835 |
| ALS | 0.780 | 0.659 | 1.438 |

Ranking/evaluation time includes generating ten full-catalog recommendations
for all 6,034 evaluated users, filtering their training interactions, and
accumulating the metrics. Data loading and preprocessing took 1.141 seconds and
is excluded from the per-model totals.

Runtime is machine- and build-dependent. It was measured on:

- Intel Core i9-9820X at 3.30 GHz, 10 physical cores / 20 logical CPUs
- Linux 6.17.0-40-generic
- Python 3.12.3
- Implicit 0.7.3, NumPy 2.5.1, SciPy 1.18.0
- CPU implementations only; native models used 10 threads
- OpenBLAS was restricted to one thread to avoid nested thread-pool overhead

## Evaluation protocol

- Read all 1,000,209 ratings from the official `ratings.dat`.
- Convert ratings of four or five stars to binary positive events.
- Retain users with at least five positive events.
- For each user, sort positive events by timestamp and hold out the latest 20%.
- Train on 457,813 events and test on 117,459 events from 6,034 users.
- Rank against all 3,706 movies present in the ratings data.
- Exclude each user's training interactions from their candidates.
- Evaluate the top 10 recommendations with seed 42.
- No test-set negative sampling is used; every non-training catalog item is a
  candidate.

The metrics are macro-averaged over users:

- Precision@10: relevant recommendations divided by 10.
- Recall@10: fraction of each user's held-out events retrieved.
- Hit Rate@10: fraction of users receiving at least one relevant result.
- MAP@10: mean truncated average precision.
- MRR@10: reciprocal rank of the first relevant result.
- NDCG@10: binary-relevance normalized discounted cumulative gain.

## Hyperparameters

| Model | Hyperparameters |
|---|---|
| Random | Per-user deterministic random ranking; `seed=42` |
| Popularity | Descending training-interaction count |
| BPR | `factors=128`, `learning_rate=0.01`, `regularization=0.01`, `iterations=3000`, `verify_negative_samples=True`, `num_threads=10`, `random_state=42` |
| NeuMF | `factors=32`, `mlp_factors=32`, `hidden_layers=(64,32,16,8)`, `learning_rate=0.001`, `regularization=1e-6`, `iterations=100`, `negative_samples=4`, `batch_size=4096`, `verify_negative_samples=True`, `inference_batch_size=16384`, `num_threads=1`, `random_state=42` |
| Item-KNN | Cosine similarity, `K=100`, `num_threads=10` |
| ALS | `factors=64`, `regularization=0.01`, `alpha=2.0`, `iterations=20`, conjugate-gradient solver, `num_threads=10`, `random_state=42` |

## Reproduction

After extracting the official `ml-1m.zip`, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=10 \
python benchmarks/benchmark_movielens_1m.py /path/to/ml-1m/ratings.dat \
  --threads 10 \
  --output benchmarks/movielens_1m_results.json
```

The benchmark implementation is in
[`benchmark_movielens_1m.py`](benchmark_movielens_1m.py), and the unrounded
measurements are stored in
[`movielens_1m_results.json`](movielens_1m_results.json).

## Limitations

This is one seeded run with a single fixed hyperparameter configuration per
model, not a hyperparameter-search study. The chronological split is stricter
than a random interaction split, but MovieLens timestamps represent rating
time rather than necessarily the time a movie was watched. Conclusions should
therefore be interpreted within this protocol. The requested iteration counts
were not chosen using a validation set; NeuMF's lower 100-epoch test metrics
show why validation-based early stopping is normally preferable.
