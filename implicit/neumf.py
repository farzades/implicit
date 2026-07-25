import numpy as np

import implicit.cpu.neumf


def NeuMF(
    factors=32,
    mlp_factors=None,
    hidden_layers=(64, 32, 16, 8),
    learning_rate=0.001,
    regularization=0.0,
    iterations=20,
    negative_samples=4,
    batch_size=1024,
    dtype=np.float32,
    use_gpu=False,
    num_threads=0,
    verify_negative_samples=True,
    inference_batch_size=65536,
    random_state=None,
):
    """Neural Matrix Factorization.

    Returns the optimized CPU NeuMF implementation. A GPU implementation is not
    currently available.

    See :class:`implicit.cpu.neumf.NeuMF` for parameter documentation.
    """
    if use_gpu:
        raise NotImplementedError("A GPU implementation of NeuMF is not available")
    return implicit.cpu.neumf.NeuMF(
        factors=factors,
        mlp_factors=mlp_factors,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        regularization=regularization,
        iterations=iterations,
        negative_samples=negative_samples,
        batch_size=batch_size,
        dtype=dtype,
        num_threads=num_threads,
        verify_negative_samples=verify_negative_samples,
        inference_batch_size=inference_batch_size,
        random_state=random_state,
    )
