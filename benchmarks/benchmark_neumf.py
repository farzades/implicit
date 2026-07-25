"""Small synthetic NeuMF training and exact-ranking benchmark."""

import argparse
import time

import numpy as np
from scipy.sparse import csr_matrix

from implicit.neumf import NeuMF


def interactions(users, items, interactions_per_user, random_state):
    userids = np.repeat(np.arange(users), interactions_per_user)
    itemids = random_state.integers(0, items, size=len(userids))
    values = np.ones(len(userids), dtype=np.float32)
    matrix = csr_matrix((values, (userids, itemids)), shape=(users, items))
    matrix.sum_duplicates()
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=10000)
    parser.add_argument("--items", type=int, default=50000)
    parser.add_argument("--interactions", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    random_state = np.random.default_rng(42)
    user_items = interactions(args.users, args.items, args.interactions, random_state)
    model = NeuMF(iterations=args.iterations, batch_size=4096, random_state=42)

    start = time.perf_counter()
    model.fit(user_items, show_progress=False)
    fit_seconds = time.perf_counter() - start

    query_users = np.arange(min(args.queries, args.users))
    start = time.perf_counter()
    model.recommend(query_users, user_items[query_users], N=10)
    recommend_seconds = time.perf_counter() - start

    examples = user_items.nnz * (model.negative_samples + 1) * model.iterations
    scored = len(query_users) * args.items
    print(f"fit: {fit_seconds:.3f}s ({examples / fit_seconds:,.0f} examples/s)")
    print(f"recommend: {recommend_seconds:.3f}s ({scored / recommend_seconds:,.0f} scores/s)")


if __name__ == "__main__":
    main()
