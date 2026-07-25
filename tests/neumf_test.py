import io
import tempfile
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from implicit.cpu.neumf import NeuMF as CPUNeuMF
from implicit.neumf import NeuMF


def checkerboard(size):
    values = np.zeros((size, size), dtype=np.float32)
    for user in range(size):
        values[user, user % 2 :: 2] = 1
        values[user, user] = 0
    return csr_matrix(values)


def small_model(**kwargs):
    parameters = dict(
        factors=8,
        mlp_factors=8,
        hidden_layers=(16, 8),
        learning_rate=0.01,
        iterations=30,
        negative_samples=2,
        batch_size=128,
        random_state=42,
    )
    parameters.update(kwargs)
    return NeuMF(**parameters)


class NeuMFTest(unittest.TestCase):
    def test_factory_and_parameter_validation(self):
        assert isinstance(NeuMF(), CPUNeuMF)
        with pytest.raises(NotImplementedError):
            NeuMF(use_gpu=True)

        invalid = (
            {"factors": 0},
            {"mlp_factors": 0},
            {"hidden_layers": ()},
            {"hidden_layers": (8, 0)},
            {"learning_rate": 0},
            {"regularization": -1},
            {"iterations": -1},
            {"negative_samples": 0},
            {"batch_size": 2, "negative_samples": 2},
            {"inference_batch_size": 0},
            {"dtype": np.float16},
        )
        for parameters in invalid:
            with pytest.raises(ValueError):
                NeuMF(**parameters)

    def test_training_loss_and_ranking(self):
        interactions = checkerboard(20)
        losses = []
        model = small_model()
        returned = model.fit(
            interactions,
            show_progress=False,
            callback=lambda epoch, elapsed, loss: losses.append(loss),
        )

        assert returned is model
        assert len(losses) == model.iterations
        assert losses[-1] < losses[0] * 0.4
        for user in range(interactions.shape[0]):
            ids, _ = model.recommend(user, interactions[user], N=1)
            assert ids[0] == user

    def test_score_predict_and_batch_recommend(self):
        interactions = checkerboard(16)
        model = small_model(iterations=20).fit(interactions, show_progress=False)

        itemids = np.arange(interactions.shape[1])
        scores = model.score(0, itemids)
        assert scores.dtype == np.float32
        assert_allclose(scores, model._score_user_items(0, itemids))
        probabilities = model.predict(0, itemids)
        assert np.all((0 <= probabilities) & (probabilities <= 1))
        assert_allclose(model.score(0, 0), scores[0], rtol=1e-6)

        userids = np.array([1, 4, 7])
        batch_ids, batch_scores = model.recommend(
            userids, interactions[userids], N=4, filter_already_liked_items=False
        )
        for row, user in enumerate(userids):
            ids, user_scores = model.recommend(
                user, interactions[user], N=4, filter_already_liked_items=False
            )
            assert_array_equal(ids, batch_ids[row])
            assert_allclose(user_scores, batch_scores[row])

    def test_filters_candidates_and_chunking(self):
        interactions = checkerboard(20)
        model = small_model(iterations=20, inference_batch_size=3).fit(
            interactions, show_progress=False
        )

        candidates = [9, 7, 5, 3, 1]
        ids, scores = model.recommend(
            0,
            interactions[0],
            N=5,
            items=candidates,
            filter_already_liked_items=False,
        )
        assert set(ids) == set(candidates)
        assert np.all(scores[:-1] >= scores[1:])

        ids, _ = model.recommend(0, interactions[0], N=2, filter_items=[0, 1, 2])
        assert not set(ids).intersection({0, 1, 2})
        assert not set(ids).intersection(interactions[0].indices)

        with pytest.raises(ValueError):
            model.recommend(0, interactions[0], items=[1], filter_items=[2])
        with pytest.raises(IndexError):
            model.recommend(0, interactions[0], items=[-1, 2])
        with pytest.raises(IndexError):
            model.score(100, 0)
        with pytest.raises(NotImplementedError):
            model.recommend(0, interactions[0], recalculate_user=True)

    def test_serialization_and_pickle_free_format(self):
        interactions = checkerboard(12)
        model = small_model(iterations=10).fit(interactions, show_progress=False)
        expected = model.score(np.array([0, 1, 2]), np.array([3, 4, 5]))

        with tempfile.NamedTemporaryFile(suffix=".npz") as output:
            model.save(output.name)
            loaded = CPUNeuMF.load(output.name)
        assert isinstance(loaded.hidden_layers, tuple)
        assert loaded.dtype == model.dtype
        assert_allclose(loaded.score(np.array([0, 1, 2]), np.array([3, 4, 5])), expected)

        output = io.BytesIO()
        model.save(output)
        output.seek(0)
        loaded = CPUNeuMF.load(output)
        assert_allclose(loaded.score(np.array([0, 1, 2]), np.array([3, 4, 5])), expected)

        unfitted = small_model(iterations=0)
        output = io.BytesIO()
        unfitted.save(output)
        output.seek(0)
        reloaded = CPUNeuMF.load(output)
        assert unfitted.__dict__ == reloaded.__dict__

    def test_reproducible_and_handles_degenerate_rows(self):
        interactions = csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=np.float64,
            )
        )
        first = small_model(dtype=np.float64, iterations=3).fit(interactions, show_progress=False)
        second = small_model(dtype=np.float64, iterations=3).fit(interactions, show_progress=False)
        assert_allclose(first.user_factors, second.user_factors, rtol=0, atol=0)
        assert_allclose(first.item_factors, second.item_factors, rtol=0, atol=0)
        assert_array_equal(first.user_factors[0], np.zeros(first.user_factors.shape[1]))
        assert np.isfinite(first.score(1, np.arange(4))).all()

        legacy_state = np.random.RandomState(42)
        model = small_model(iterations=0, random_state=legacy_state)
        model.fit(interactions, show_progress=False)
        assert model.user_factors.shape[0] == interactions.shape[0]

    def test_backpropagation_matches_finite_differences(self):
        interactions = checkerboard(4).astype(np.float64)
        model = small_model(
            dtype=np.float64,
            factors=2,
            mlp_factors=2,
            hidden_layers=(3,),
            iterations=0,
        ).fit(interactions, show_progress=False)
        model.mlp_biases.fill(1.0)  # keep ReLU away from its nondifferentiable point

        users = np.array([0, 1], dtype=np.intp)
        items = np.array([1, 2], dtype=np.intp)
        labels = np.array([1.0, 0.0])
        gradients = {}

        def capture_sparse(values, gradient, ids, first, second, step_size):
            del ids, first, second, step_size
            gradients[id(values)] = gradient.copy()

        def capture_dense(value, gradient, first, second, step_size):
            del first, second, step_size
            gradients[id(value)] = gradient.copy()

        model._adam_sparse = capture_sparse
        model._adam_dense = capture_dense
        output_bias = model.output_bias
        model._train_batch(users, items, labels)
        model.output_bias = output_bias

        def loss():
            gmf = model._user_mf[users] * model._item_mf[items]
            activation = np.concatenate((model._user_mlp[users], model._item_mlp[items]), axis=1)
            activation = activation @ model._weight_views()[0] + model._bias_views()[0]
            activation = np.maximum(activation, 0)
            logits = (
                gmf @ model.output_weights[: model.factors]
                + activation @ model.output_weights[model.factors :]
                + model.output_bias
            )
            return np.mean(np.logaddexp(0, logits) - labels * logits)

        epsilon = 1e-6
        checks = (
            (model._user_mf, (0, 0), gradients[id(model._user_mf)][0, 0]),
            (model._item_mlp, (1, 1), gradients[id(model._item_mlp)][0, 1]),
            (model.mlp_weights, (2,), gradients[id(model.mlp_weights)][2]),
            (model.output_weights, (0,), gradients[id(model.output_weights)][0]),
        )
        for parameter, index, analytic in checks:
            original = parameter[index]
            parameter[index] = original + epsilon
            upper = loss()
            parameter[index] = original - epsilon
            lower = loss()
            parameter[index] = original
            assert_allclose(analytic, (upper - lower) / (2 * epsilon), rtol=1e-5, atol=1e-8)
