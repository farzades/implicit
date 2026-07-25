"""Benchmark ranking quality of Implicit recommenders on MovieLens-1M.

The script expects the official MovieLens-1M ``ratings.dat`` file. Ratings of
four or five stars are converted to binary implicit events. For every user with
at least five positive events, the latest 20 percent are held out for testing.
All metrics are calculated against the full rated-item catalog.
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from implicit.neumf import NeuMF

SEED = 42
K = 10
MODEL_ORDER = ("Random", "Popularity", "BPR", "NeuMF", "Item-KNN", "ALS")


def load_data(path, rating_threshold=4, minimum_positives=5, test_fraction=0.2):
    users = []
    items = []
    ratings = []
    timestamps = []
    with open(path, encoding="latin-1") as ratings_file:
        for line in ratings_file:
            user, item, rating, timestamp = map(int, line.rstrip().split("::"))
            users.append(user)
            items.append(item)
            ratings.append(rating)
            timestamps.append(timestamp)

    users = np.asarray(users, dtype=np.int32)
    items = np.asarray(items, dtype=np.int32)
    ratings = np.asarray(ratings, dtype=np.int8)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    item_ids = np.unique(items)

    positive = ratings >= rating_threshold
    positive_users = users[positive]
    unique_users, counts = np.unique(positive_users, return_counts=True)
    eligible_users = unique_users[counts >= minimum_positives]

    user_lookup = np.full(users.max() + 1, -1, dtype=np.int32)
    user_lookup[eligible_users] = np.arange(len(eligible_users), dtype=np.int32)
    selected = positive & (user_lookup[users] >= 0)
    dense_users = user_lookup[users[selected]]
    dense_items = np.searchsorted(item_ids, items[selected]).astype(np.int32)
    positive_timestamps = timestamps[selected]

    order = np.lexsort((dense_items, positive_timestamps, dense_users))
    dense_users = dense_users[order]
    dense_items = dense_items[order]
    counts = np.bincount(dense_users, minlength=len(eligible_users))
    starts = np.r_[0, np.cumsum(counts)]

    test_mask = np.zeros(len(dense_users), dtype=bool)
    for user, count in enumerate(counts):
        test_count = max(1, int(np.ceil(count * test_fraction)))
        test_mask[starts[user + 1] - test_count : starts[user + 1]] = True

    shape = (len(eligible_users), len(item_ids))
    train = csr_matrix(
        (
            np.ones(np.count_nonzero(~test_mask), dtype=np.float32),
            (dense_users[~test_mask], dense_items[~test_mask]),
        ),
        shape=shape,
    )
    test = csr_matrix(
        (
            np.ones(np.count_nonzero(test_mask), dtype=np.float32),
            (dense_users[test_mask], dense_items[test_mask]),
        ),
        shape=shape,
    )
    train.sort_indices()
    test.sort_indices()

    statistics = {
        "raw_ratings": int(len(ratings)),
        "raw_users": int(len(np.unique(users))),
        "catalog_items": int(len(item_ids)),
        "positive_events": int(np.count_nonzero(positive)),
        "evaluated_users": int(shape[0]),
        "train_events": int(train.nnz),
        "test_events": int(test.nnz),
        "rating_threshold": int(rating_threshold),
        "minimum_positives": int(minimum_positives),
        "test_fraction": float(test_fraction),
    }
    return train, test, statistics


def top_k(scores, count):
    selected = np.argpartition(scores, -count)[-count:]
    order = np.lexsort((selected, -scores[selected]))
    selected = selected[order]
    return selected.astype(np.int32), scores[selected]


class RandomRecommender:
    def __init__(self, seed=SEED):
        self.seed = seed
        self.items = 0

    def fit(self, user_items, show_progress=False):
        del show_progress
        self.items = user_items.shape[1]
        return self

    def recommend(self, userid, user_items, N=10, **kwargs):
        del kwargs
        scalar = np.isscalar(userid)
        userids = np.atleast_1d(userid)
        result_ids = np.empty((len(userids), N), dtype=np.int32)
        result_scores = np.empty((len(userids), N), dtype=np.float64)
        for row, user in enumerate(userids):
            random_state = np.random.default_rng(self.seed + int(user))
            scores = random_state.random(self.items)
            start, end = user_items.indptr[row : row + 2]
            scores[user_items.indices[start:end]] = -np.inf
            result_ids[row], result_scores[row] = top_k(scores, N)
        if scalar:
            return result_ids[0], result_scores[0]
        return result_ids, result_scores


class PopularityRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, user_items, show_progress=False):
        del show_progress
        self.popularity = np.asarray(user_items.sum(axis=0)).reshape(-1).astype(np.float64)
        return self

    def recommend(self, userid, user_items, N=10, **kwargs):
        del kwargs
        scalar = np.isscalar(userid)
        userids = np.atleast_1d(userid)
        result_ids = np.empty((len(userids), N), dtype=np.int32)
        result_scores = np.empty((len(userids), N), dtype=np.float64)
        for row in range(len(userids)):
            scores = self.popularity.copy()
            start, end = user_items.indptr[row : row + 2]
            scores[user_items.indices[start:end]] = -np.inf
            result_ids[row], result_scores[row] = top_k(scores, N)
        if scalar:
            return result_ids[0], result_scores[0]
        return result_ids, result_scores


def build_models(num_threads):
    return {
        "Random": (
            RandomRecommender(seed=SEED),
            {"seed": SEED},
        ),
        "Popularity": (
            PopularityRecommender(),
            {"score": "training interaction count"},
        ),
        "BPR": (
            BayesianPersonalizedRanking(
                factors=128,
                learning_rate=0.01,
                regularization=0.01,
                iterations=3000,
                verify_negative_samples=True,
                num_threads=num_threads,
                random_state=SEED,
                use_gpu=False,
            ),
            {
                "factors": 128,
                "learning_rate": 0.01,
                "regularization": 0.01,
                "iterations": 3000,
                "verify_negative_samples": True,
                "num_threads": num_threads,
                "random_state": SEED,
            },
        ),
        "NeuMF": (
            NeuMF(
                factors=32,
                mlp_factors=32,
                hidden_layers=(64, 32, 16, 8),
                learning_rate=0.001,
                regularization=1e-6,
                iterations=100,
                negative_samples=4,
                batch_size=4096,
                verify_negative_samples=True,
                inference_batch_size=16384,
                num_threads=1,
                random_state=SEED,
            ),
            {
                "factors": 32,
                "mlp_factors": 32,
                "hidden_layers": [64, 32, 16, 8],
                "learning_rate": 0.001,
                "regularization": 1e-6,
                "iterations": 100,
                "negative_samples": 4,
                "batch_size": 4096,
                "verify_negative_samples": True,
                "inference_batch_size": 16384,
                "num_threads": 1,
                "random_state": SEED,
            },
        ),
        "Item-KNN": (
            CosineRecommender(K=100, num_threads=num_threads),
            {
                "similarity": "cosine",
                "neighbors": 100,
                "num_threads": num_threads,
            },
        ),
        "ALS": (
            AlternatingLeastSquares(
                factors=64,
                regularization=0.01,
                alpha=2.0,
                iterations=20,
                use_cg=True,
                num_threads=num_threads,
                random_state=SEED,
                use_gpu=False,
            ),
            {
                "factors": 64,
                "regularization": 0.01,
                "alpha": 2.0,
                "iterations": 20,
                "use_cg": True,
                "num_threads": num_threads,
                "random_state": SEED,
            },
        ),
    }


def ranking_metrics(model, train, test, k=K, batch_size=500):
    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "hit_rate": 0.0,
        "map": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
    users = np.flatnonzero(np.diff(test.indptr)).astype(np.int32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for offset in range(0, len(users), batch_size):
        batch = users[offset : offset + batch_size]
        recommendations, _ = model.recommend(batch, train[batch], N=k)
        for row, user in enumerate(batch):
            truth = test.indices[test.indptr[user] : test.indptr[user + 1]]
            hits = np.isin(recommendations[row], truth, assume_unique=False)
            hit_count = int(np.count_nonzero(hits))
            cumulative_hits = np.cumsum(hits)
            ranks = np.flatnonzero(hits)

            totals["precision"] += hit_count / k
            totals["recall"] += hit_count / len(truth)
            totals["hit_rate"] += bool(hit_count)
            totals["map"] += np.sum(cumulative_hits[hits] / (ranks + 1)) / min(len(truth), k)
            totals["mrr"] += 0.0 if not len(ranks) else 1.0 / (ranks[0] + 1)
            ideal = np.sum(discounts[: min(len(truth), k)])
            totals["ndcg"] += np.sum(discounts[hits]) / ideal

    return {f"{name}@{k}": value / len(users) for name, value in totals.items()}


def environment():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "implicit": implicit.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", type=Path, help="Path to MovieLens-1M ratings.dat")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--output", type=Path, default=Path("movielens_1m_results.json"))
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    requested = args.models.split(",")
    unknown = set(requested) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    data_start = time.perf_counter()
    train, test, statistics = load_data(args.ratings)
    data_seconds = time.perf_counter() - data_start
    print(f"data: {statistics} ({data_seconds:.3f}s)", flush=True)

    payload = {
        "protocol": {
            "dataset": "MovieLens 1M",
            "positive_rating_threshold": "rating >= 4",
            "split": "per-user chronological; latest 20% held out",
            "minimum_positive_events": 5,
            "candidate_set": "all 3,706 rated movies, excluding training interactions",
            "k": K,
            "seed": SEED,
        },
        "data": statistics,
        "environment": environment(),
        "data_preparation_seconds": data_seconds,
        "models": {},
    }
    if args.output.exists():
        with open(args.output, encoding="utf8") as results_file:
            previous = json.load(results_file)
        payload["models"].update(previous.get("models", {}))

    models = build_models(args.threads)
    for name in MODEL_ORDER:
        if name not in requested:
            continue
        model, hyperparameters = models[name]
        print(f"{name}: fitting", flush=True)
        start = time.perf_counter()
        model.fit(train, show_progress=False)
        fit_seconds = time.perf_counter() - start

        print(f"{name}: evaluating", flush=True)
        start = time.perf_counter()
        metrics = ranking_metrics(model, train, test)
        evaluation_seconds = time.perf_counter() - start
        payload["models"][name] = {
            "hyperparameters": hyperparameters,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
            "metrics": metrics,
        }
        with open(args.output, "w", encoding="utf8") as results_file:
            json.dump(payload, results_file, indent=2)
            results_file.write("\n")
        print(
            f"{name}: fit={fit_seconds:.3f}s evaluation={evaluation_seconds:.3f}s "
            f"metrics={metrics}",
            flush=True,
        )


if __name__ == "__main__":
    main()
