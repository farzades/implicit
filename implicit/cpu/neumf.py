"""CPU implementation of Neural Matrix Factorization (NeuMF)."""

import logging
import time
from contextlib import nullcontext

import numpy as np
import threadpoolctl
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from ..recommender_base import RecommenderBase
from ..utils import check_csr, check_random_state
from .matrix_factorization_base import MatrixFactorizationBase

log = logging.getLogger("implicit")


class NeuMF(MatrixFactorizationBase):
    """Neural Matrix Factorization.

    NeuMF combines a generalized matrix-factorization (GMF) path with a multi-layer
    perceptron (MLP) path. The two paths use separate user and item embeddings and
    are joined by a learned output layer. Training uses binary cross-entropy and
    uniformly sampled unobserved user-item pairs.

    This is the NeuMF model introduced in `Neural Collaborative Filtering
    <https://arxiv.org/abs/1708.05031>`_. The later `Neural Collaborative Ranking
    <https://arxiv.org/abs/1808.04957>`_ paper evaluates NeuMF as a pointwise
    baseline; its proposed pairwise fusion model is named NeuPR.

    Parameters
    ----------
    factors : int, optional
        Embedding size of the GMF path.
    mlp_factors : int, optional
        User and item embedding size of the MLP path. Defaults to ``factors``.
    hidden_layers : sequence of int, optional
        Width of each MLP hidden layer.
    learning_rate : float, optional
        Adam learning rate.
    regularization : float, optional
        L2 regularization applied to embeddings and dense weights.
    iterations : int, optional
        Number of training epochs.
    negative_samples : int, optional
        Number of uniformly sampled unobserved interactions per positive interaction.
    batch_size : int, optional
        Maximum number of positive and negative examples in a training batch.
    dtype : data-type, optional
        ``numpy.float32`` or ``numpy.float64``. Float32 is normally faster and uses
        half the memory.
    num_threads : int, optional
        Number of BLAS threads used while fitting, and native threads used by the
        inherited similarity methods. Zero uses the library default.
    verify_negative_samples : bool, optional
        Resample a negative pair if it is present in the interaction matrix.
    inference_batch_size : int, optional
        Maximum number of candidate items scored at once. Lower values reduce peak
        inference memory, while higher values can improve BLAS throughput.
    random_state : int, RandomState, Generator or None, optional
        Random seed or random state used for initialization, shuffling, and sampling.

    Attributes
    ----------
    item_factors : ndarray
        Concatenated GMF and MLP item embeddings, used by ``similar_items``.
    user_factors : ndarray
        Concatenated GMF and MLP user embeddings, used by ``similar_users``.

    Notes
    -----
    Recommendation scores are logits. Applying a sigmoid changes their calibration,
    but not their ranking.
    """

    def __init__(
        self,
        factors=32,
        mlp_factors=None,
        hidden_layers=(64, 32, 16, 8),
        learning_rate=0.001,
        regularization=0.0,
        iterations=20,
        negative_samples=4,
        batch_size=1024,
        dtype=np.float32,
        num_threads=0,
        verify_negative_samples=True,
        inference_batch_size=65536,
        random_state=None,
    ):
        super().__init__(num_threads=num_threads)

        if mlp_factors is None:
            mlp_factors = factors

        self.factors = int(factors)
        self.mlp_factors = int(mlp_factors)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.negative_samples = int(negative_samples)
        self.batch_size = int(batch_size)
        self.dtype = np.dtype(dtype)
        self.verify_negative_samples = bool(verify_negative_samples)
        self.inference_batch_size = int(inference_batch_size)
        self.random_state = random_state
        self._validate_parameters()

        self._user_mf = None
        self._item_mf = None
        self._user_mlp = None
        self._item_mlp = None
        self.mlp_weights = None
        self.mlp_biases = None
        self.output_weights = None
        self.output_bias = 0.0

        # Adam state is retained to support efficient warm-started fit calls.
        self._optimizer_step = 0
        for name in self._parameter_names():
            setattr(self, f"{name}_m", None)
            setattr(self, f"{name}_v", None)
        self._output_bias_m = 0.0
        self._output_bias_v = 0.0

    @staticmethod
    def _parameter_names():
        return (
            "_user_mf",
            "_item_mf",
            "_user_mlp",
            "_item_mlp",
            "mlp_weights",
            "mlp_biases",
            "output_weights",
        )

    def _validate_parameters(self):
        if self.factors <= 0:
            raise ValueError("factors must be greater than zero")
        if self.mlp_factors <= 0:
            raise ValueError("mlp_factors must be greater than zero")
        if not self.hidden_layers or any(width <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive layer widths")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.negative_samples <= 0:
            raise ValueError("negative_samples must be greater than zero")
        if self.batch_size < self.negative_samples + 1:
            raise ValueError("batch_size must fit one positive and its negative samples")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be greater than zero")
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be numpy.float32 or numpy.float64")

    def _layer_shapes(self):
        inputs = (2 * self.mlp_factors,) + tuple(self.hidden_layers[:-1])
        return tuple(zip(inputs, self.hidden_layers))

    def _weight_views(self, values=None):
        values = self.mlp_weights if values is None else values
        offset = 0
        views = []
        for inputs, outputs in self._layer_shapes():
            size = inputs * outputs
            views.append(values[offset : offset + size].reshape(inputs, outputs))
            offset += size
        return views

    def _bias_views(self, values=None):
        values = self.mlp_biases if values is None else values
        offset = 0
        views = []
        for width in self.hidden_layers:
            views.append(values[offset : offset + width])
            offset += width
        return views

    def _initialize(self, users, items, user_counts, item_counts, random_state):
        def normal(shape):
            return random_state.normal(0.0, 0.01, size=shape).astype(self.dtype)

        self._user_mf = normal((users, self.factors))
        self._item_mf = normal((items, self.factors))
        self._user_mlp = normal((users, self.mlp_factors))
        self._item_mlp = normal((items, self.mlp_factors))

        # Match the rest of implicit: absent entities have zero representations.
        self._user_mf[user_counts == 0] = 0
        self._user_mlp[user_counts == 0] = 0
        self._item_mf[item_counts == 0] = 0
        self._item_mlp[item_counts == 0] = 0

        weights = []
        for fan_in, fan_out in self._layer_shapes():
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(
                random_state.uniform(-limit, limit, size=fan_in * fan_out).astype(self.dtype)
            )
        self.mlp_weights = np.concatenate(weights)
        self.mlp_biases = np.zeros(sum(self.hidden_layers), dtype=self.dtype)

        output_size = self.factors + self.hidden_layers[-1]
        limit = np.sqrt(6.0 / (output_size + 1))
        self.output_weights = random_state.uniform(-limit, limit, size=output_size).astype(
            self.dtype
        )
        self.output_bias = 0.0
        self._reset_optimizer()
        self._sync_factors()

    def _reset_optimizer(self):
        self._optimizer_step = 0
        for name in self._parameter_names():
            value = getattr(self, name)
            setattr(self, f"{name}_m", np.zeros_like(value))
            setattr(self, f"{name}_v", np.zeros_like(value))
        self._output_bias_m = 0.0
        self._output_bias_v = 0.0

    def _ensure_optimizer(self):
        for name in self._parameter_names():
            value = getattr(self, name)
            first = getattr(self, f"{name}_m", None)
            second = getattr(self, f"{name}_v", None)
            if first is None or first.shape != value.shape:
                setattr(self, f"{name}_m", np.zeros_like(value))
            if second is None or second.shape != value.shape:
                setattr(self, f"{name}_v", np.zeros_like(value))

    def _sync_factors(self):
        self.user_factors = np.ascontiguousarray(
            np.concatenate((self._user_mf, self._user_mlp), axis=1)
        )
        self.item_factors = np.ascontiguousarray(
            np.concatenate((self._item_mf, self._item_mlp), axis=1)
        )
        self._user_norms = self._item_norms = None

    @staticmethod
    def _sigmoid(values):
        values = np.asarray(values)
        output = np.empty_like(values)
        positive = values >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output

    @staticmethod
    def _adam_dense(value, gradient, first, second, step_size):
        first *= 0.9
        first += 0.1 * gradient
        second *= 0.999
        second += 0.001 * gradient * gradient
        value -= step_size * first / (np.sqrt(second) + 1e-8)

    def _adam_sparse(self, values, gradient, ids, first, second, step_size):
        # Sorting plus reduceat is much faster than np.add.at for repeated ids.
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        unique = sorted_ids[starts]
        aggregated = np.add.reduceat(gradient[order], starts, axis=0)
        if self.regularization:
            aggregated += self.regularization * values[unique]

        first_rows = 0.9 * first[unique] + 0.1 * aggregated
        second_rows = 0.999 * second[unique] + 0.001 * aggregated * aggregated
        values[unique] -= step_size * first_rows / (np.sqrt(second_rows) + 1e-8)
        first[unique] = first_rows
        second[unique] = second_rows

    def _train_batch(self, users, items, labels):
        user_mf = self._user_mf[users]
        item_mf = self._item_mf[items]
        user_mlp = self._user_mlp[users]
        item_mlp = self._item_mlp[items]

        gmf = user_mf * item_mf
        activation = np.concatenate((user_mlp, item_mlp), axis=1)
        activations = [activation]
        weight_views = self._weight_views()
        for weights, bias in zip(weight_views, self._bias_views()):
            activation = activation @ weights + bias
            np.maximum(activation, 0, out=activation)
            activations.append(activation)

        output_gmf = self.output_weights[: self.factors]
        output_mlp = self.output_weights[self.factors :]
        logits = gmf @ output_gmf + activations[-1] @ output_mlp + self.output_bias
        loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)

        derivative = (self._sigmoid(logits) - labels) / len(labels)
        output_gradient = np.concatenate((gmf.T @ derivative, activations[-1].T @ derivative))
        if self.regularization:
            output_gradient += self.regularization * self.output_weights
        output_bias_gradient = float(np.sum(derivative))

        # Compute all gradients before changing any parameter.
        delta = derivative[:, None] * output_mlp[None, :]
        weight_gradients = [None] * len(weight_views)
        bias_gradients = [None] * len(weight_views)
        input_gradient = None
        for layer in range(len(weight_views) - 1, -1, -1):
            delta *= activations[layer + 1] > 0
            weights = weight_views[layer]
            weight_gradient = activations[layer].T @ delta
            if self.regularization:
                weight_gradient += self.regularization * weights
            weight_gradients[layer] = weight_gradient
            bias_gradients[layer] = np.sum(delta, axis=0)
            propagated = delta @ weights.T
            if layer:
                delta = propagated
            else:
                input_gradient = propagated

        user_mf_gradient = derivative[:, None] * output_gmf[None, :] * item_mf
        item_mf_gradient = derivative[:, None] * output_gmf[None, :] * user_mf
        user_mlp_gradient = input_gradient[:, : self.mlp_factors]
        item_mlp_gradient = input_gradient[:, self.mlp_factors :]

        self._optimizer_step += 1
        correction = (
            self.learning_rate
            * np.sqrt(1.0 - 0.999**self._optimizer_step)
            / (1.0 - 0.9**self._optimizer_step)
        )

        for values, gradient, ids, first, second in (
            (self._user_mf, user_mf_gradient, users, self._user_mf_m, self._user_mf_v),
            (self._item_mf, item_mf_gradient, items, self._item_mf_m, self._item_mf_v),
            (self._user_mlp, user_mlp_gradient, users, self._user_mlp_m, self._user_mlp_v),
            (self._item_mlp, item_mlp_gradient, items, self._item_mlp_m, self._item_mlp_v),
        ):
            self._adam_sparse(values, gradient, ids, first, second, correction)

        flat_weight_gradient = np.concatenate(
            [gradient.reshape(-1) for gradient in weight_gradients]
        )
        flat_bias_gradient = np.concatenate(bias_gradients)
        for value, gradient, first, second in (
            (
                self.mlp_weights,
                flat_weight_gradient,
                self.mlp_weights_m,
                self.mlp_weights_v,
            ),
            (self.mlp_biases, flat_bias_gradient, self.mlp_biases_m, self.mlp_biases_v),
            (
                self.output_weights,
                output_gradient,
                self.output_weights_m,
                self.output_weights_v,
            ),
        ):
            self._adam_dense(value, gradient, first, second, correction)

        self._output_bias_m = 0.9 * self._output_bias_m + 0.1 * output_bias_gradient
        self._output_bias_v = (
            0.999 * self._output_bias_v + 0.001 * output_bias_gradient * output_bias_gradient
        )
        self.output_bias -= correction * self._output_bias_m / (np.sqrt(self._output_bias_v) + 1e-8)
        return float(loss)

    @staticmethod
    def _contains_sorted(sorted_values, values):
        positions = np.searchsorted(sorted_values, values)
        within = positions < len(sorted_values)
        result = np.zeros(len(values), dtype=bool)
        result[within] = sorted_values[positions[within]] == values[within]
        return result

    def _sample_negatives(self, users, item_count, positive_codes, random_state):
        items = random_state.integers(0, item_count, size=len(users), dtype=np.intp)
        if not self.verify_negative_samples or not len(positive_codes):
            return items

        codes = users.astype(np.int64, copy=False) * item_count + items
        collisions = self._contains_sorted(positive_codes, codes)
        while np.any(collisions):
            items[collisions] = random_state.integers(
                0, item_count, size=np.count_nonzero(collisions), dtype=np.intp
            )
            codes[collisions] = users[collisions] * item_count + items[collisions]
            collisions = self._contains_sorted(positive_codes, codes)
        return items

    def fit(self, user_items, show_progress=True, callback=None):
        """Fit the model to a sparse user-item interaction matrix.

        Every stored nonzero is treated as a positive event; its numeric value is
        ignored. Unobserved entries are sampled as negatives according to
        ``negative_samples``.
        """
        random_state = check_random_state(self.random_state)
        user_items = check_csr(user_items)
        if user_items.dtype != self.dtype:
            user_items = user_items.astype(self.dtype)
        else:
            user_items = user_items.copy()
        user_items.sum_duplicates()
        user_items.eliminate_zeros()
        if not user_items.has_sorted_indices:
            user_items.sort_indices()

        users, items = user_items.shape
        if items == 0:
            raise ValueError("user_items must contain at least one item")
        user_counts = np.diff(user_items.indptr)
        item_counts = np.bincount(user_items.indices, minlength=items)

        current_shape = (
            None
            if self._user_mf is None
            else (
                self._user_mf.shape[0],
                self._item_mf.shape[0],
            )
        )
        if current_shape != (users, items):
            self._initialize(users, items, user_counts, item_counts, random_state)
        else:
            self._ensure_optimizer()

        positive_users = np.repeat(np.arange(users, dtype=np.intp), user_counts)
        positive_items = user_items.indices.astype(np.intp, copy=False)

        # A fully observed user has no valid negative item.
        trainable = user_counts[positive_users] < items
        positive_users = positive_users[trainable]
        positive_items = positive_items[trainable]
        positive_codes = np.sort(
            positive_users.astype(np.int64) * items + positive_items.astype(np.int64)
        )

        positives_per_batch = max(1, self.batch_size // (self.negative_samples + 1))
        blas_context = (
            threadpoolctl.threadpool_limits(self.num_threads, user_api="blas")
            if self.num_threads
            else nullcontext()
        )

        log.debug("Running %i NeuMF training epochs", self.iterations)
        with blas_context, tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                start = time.time()
                order = random_state.permutation(len(positive_users))
                total_loss = 0.0
                total_examples = 0
                for offset in range(0, len(order), positives_per_batch):
                    selected = order[offset : offset + positives_per_batch]
                    batch_positive_users = positive_users[selected]
                    batch_positive_items = positive_items[selected]
                    negative_users = np.repeat(batch_positive_users, self.negative_samples)
                    negative_items = self._sample_negatives(
                        negative_users, items, positive_codes, random_state
                    )

                    batch_users = np.concatenate((batch_positive_users, negative_users))
                    batch_items = np.concatenate((batch_positive_items, negative_items))
                    labels = np.concatenate(
                        (
                            np.ones(len(batch_positive_users), dtype=self.dtype),
                            np.zeros(len(negative_users), dtype=self.dtype),
                        )
                    )
                    batch_order = random_state.permutation(len(labels))
                    batch_loss = self._train_batch(
                        batch_users[batch_order],
                        batch_items[batch_order],
                        labels[batch_order],
                    )
                    total_loss += batch_loss * len(labels)
                    total_examples += len(labels)

                loss = total_loss / total_examples if total_examples else 0.0
                self._sync_factors()
                progress.update(1)
                progress.set_postfix({"loss": f"{loss:.4f}"})
                if callback:
                    callback(epoch, time.time() - start, loss)

        self._sync_factors()
        self._check_fit_errors()
        return self

    def _score_user_items(
        self,
        userid,
        itemids,
        item_projection=None,
        user_projection=None,
        gmf_user=None,
    ):
        itemids = np.asarray(itemids, dtype=np.intp)
        output_gmf = self.output_weights[: self.factors]
        if gmf_user is None:
            gmf_user = self._user_mf[userid] * output_gmf
        scores = self._item_mf[itemids] @ gmf_user

        weights = self._weight_views()
        biases = self._bias_views()
        first = weights[0]
        if item_projection is None:
            item_projection = self._item_mlp[itemids] @ first[self.mlp_factors :]
        if user_projection is None:
            user_projection = self._user_mlp[userid] @ first[: self.mlp_factors] + biases[0]
        activation = item_projection + user_projection
        np.maximum(activation, 0, out=activation)
        for layer in range(1, len(weights)):
            activation = activation @ weights[layer] + biases[layer]
            np.maximum(activation, 0, out=activation)
        scores += activation @ self.output_weights[self.factors :]
        scores += self.output_bias
        return scores

    def score(self, userid, itemid):
        """Return NeuMF logits for one pair or aligned arrays of user-item pairs."""
        if self._user_mf is None:
            raise ValueError("model has not been fit")
        scalar = np.isscalar(userid) and np.isscalar(itemid)
        users, items = np.broadcast_arrays(
            np.asarray(userid, dtype=np.intp), np.asarray(itemid, dtype=np.intp)
        )
        if np.any(users < 0) or np.any(users >= self._user_mf.shape[0]):
            raise IndexError("Some userids are not in the model")
        if np.any(items < 0) or np.any(items >= self._item_mf.shape[0]):
            raise IndexError("Some itemids are not in the model")

        flat_users = users.reshape(-1)
        flat_items = items.reshape(-1)
        result = np.empty(len(flat_users), dtype=self.dtype)
        for user in np.unique(flat_users):
            positions = np.flatnonzero(flat_users == user)
            result[positions] = self._score_user_items(user, flat_items[positions])
        result = result.reshape(users.shape)
        return result.item() if scalar else result

    def predict(self, userid, itemid):
        """Return interaction probabilities for user-item pairs."""
        scores = np.asarray(self.score(userid, itemid))
        probabilities = self._sigmoid(scores)
        return probabilities.item() if probabilities.shape == () else probabilities

    @staticmethod
    def _select_topk(ids, scores, count):
        if count < len(scores):
            selected = np.argpartition(scores, -count)[-count:]
            ids = ids[selected]
            scores = scores[selected]
        order = np.lexsort((ids, -scores))
        return ids[order][:count], scores[order][:count]

    def _recommend_users(self, userids, candidates, count, liked_items, filtered):
        weights = self._weight_views()
        biases = self._bias_views()
        first = weights[0]
        user_projections = self._user_mlp[userids] @ first[: self.mlp_factors] + biases[0]
        gmf_users = self._user_mf[userids] * self.output_weights[: self.factors]

        best_ids = [np.empty(0, dtype=np.intp) for _ in userids]
        best_scores = [np.empty(0, dtype=self.dtype) for _ in userids]

        for offset in range(0, len(candidates), self.inference_batch_size):
            itemids = candidates[offset : offset + self.inference_batch_size]
            item_projection = self._item_mlp[itemids] @ first[self.mlp_factors :]
            chunk_count = min(count, len(itemids))
            for row, user in enumerate(userids):
                scores = self._score_user_items(
                    user,
                    itemids,
                    item_projection=item_projection,
                    user_projection=user_projections[row],
                    gmf_user=gmf_users[row],
                )
                if liked_items[row] is not None and len(liked_items[row]):
                    scores[self._contains_sorted(liked_items[row], itemids)] = -np.inf
                if filtered is not None and len(filtered):
                    scores[self._contains_sorted(filtered, itemids)] = -np.inf

                chunk_ids, chunk_scores = self._select_topk(itemids, scores, chunk_count)
                if len(best_ids[row]):
                    chunk_ids = np.concatenate((best_ids[row], chunk_ids))
                    chunk_scores = np.concatenate((best_scores[row], chunk_scores))
                best_ids[row], best_scores[row] = self._select_topk(
                    chunk_ids, chunk_scores, min(count, len(chunk_ids))
                )
        return np.asarray(best_ids), np.asarray(best_scores)

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        """Recommend top-ranked items using exact NeuMF scores."""
        if self._user_mf is None:
            raise ValueError("model has not been fit")
        if recalculate_user:
            raise NotImplementedError("recalculate_user is not supported with NeuMF")
        if N < 0:
            raise ValueError("N must be non-negative")
        if items is not None and filter_items is not None:
            raise ValueError("Can't set both items and filter_items in recommend call")

        scalar = np.isscalar(userid)
        userids = np.atleast_1d(np.asarray(userid, dtype=np.intp))
        if userids.ndim != 1:
            raise ValueError("userid must be a scalar or one-dimensional array")
        if np.any(userids < 0) or np.any(userids >= self._user_mf.shape[0]):
            raise IndexError("Some userids are not in the model")

        if filter_already_liked_items:
            if not isinstance(user_items, csr_matrix):
                raise ValueError("user_items needs to be a CSR sparse matrix")
            if user_items.shape[0] != len(userids):
                raise ValueError("user_items must contain 1 row for every user in userids")

        item_count = self._item_mf.shape[0]
        if items is None:
            candidates = np.arange(item_count, dtype=np.intp)
        else:
            candidates = np.sort(np.asarray(items, dtype=np.intp))
            if candidates.ndim != 1:
                raise ValueError("items must be one-dimensional")
            if len(candidates) and (candidates[0] < 0 or candidates[-1] >= item_count):
                raise IndexError("Some itemids in the items parameter are not in the model")

        count = min(N, len(candidates))
        if not count:
            shape = (len(userids), 0)
            result_ids = np.empty(shape, dtype=np.intp)
            result_scores = np.empty(shape, dtype=self.dtype)
            return (result_ids[0], result_scores[0]) if scalar else (result_ids, result_scores)

        filtered = None
        if filter_items is not None:
            filtered = np.unique(np.asarray(filter_items, dtype=np.intp))
            filtered = filtered[(filtered >= 0) & (filtered < item_count)]

        liked_items = [None] * len(userids)
        if filter_already_liked_items:
            for row in range(len(userids)):
                liked_items[row] = np.unique(
                    user_items.indices[user_items.indptr[row] : user_items.indptr[row + 1]]
                )
        result_ids, result_scores = self._recommend_users(
            userids, candidates, count, liked_items, filtered
        )

        if scalar:
            return result_ids[0], result_scores[0]
        return result_ids, result_scores

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def save(self, fileobj_or_path):
        """Save the model and Adam state in NumPy's ``.npz`` format."""
        args = {
            "factors": self.factors,
            "mlp_factors": self.mlp_factors,
            "hidden_layers": np.asarray(self.hidden_layers, dtype=np.int64),
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "negative_samples": self.negative_samples,
            "batch_size": self.batch_size,
            "dtype": self.dtype.name,
            "num_threads": self.num_threads,
            "verify_negative_samples": self.verify_negative_samples,
            "inference_batch_size": self.inference_batch_size,
            "_optimizer_step": self._optimizer_step,
            "output_bias": self.output_bias,
            "_output_bias_m": self._output_bias_m,
            "_output_bias_v": self._output_bias_v,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
        }
        for name in self._parameter_names():
            args[name] = getattr(self, name)
            args[f"{name}_m"] = getattr(self, f"{name}_m")
            args[f"{name}_v"] = getattr(self, f"{name}_v")
        if isinstance(self.random_state, (int, np.integer)):
            args["random_state"] = int(self.random_state)

        args = {key: value for key, value in args.items() if value is not None}
        np.savez(fileobj_or_path, **args)

    @classmethod
    def load(cls, fileobj_or_path):
        """Load a NeuMF model saved with :meth:`save`."""
        if isinstance(fileobj_or_path, str) and not fileobj_or_path.endswith(".npz"):
            fileobj_or_path = fileobj_or_path + ".npz"
        with np.load(fileobj_or_path, allow_pickle=False) as data:
            model = cls()
            for name, value in data.items():
                if name == "dtype":
                    value = np.dtype(str(value))
                elif name == "hidden_layers":
                    value = tuple(int(width) for width in value)
                elif value.shape == ():
                    value = value.item()
                setattr(model, name, value)
        return model
