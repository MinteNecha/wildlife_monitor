"""
Gradient checks for the hand-written autodiff engine.

This is the most load-bearing test in the project. Every behavioural model is
trained by these gradients, and a wrong backward pass does not crash — it
quietly trains the wrong thing. Each test compares the analytical gradient
against a central finite difference, which is an independent computation of
the same quantity.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_monitor.pipeline2.autodiff import (
    Tensor, softmax_cross_entropy, clip_grad_norm, layernorm, Adam,
)

TOLERANCE = 1e-6


def numerical_gradient(function, tensor: Tensor, epsilon: float = 1e-6):
    """Central finite difference of ``function`` with respect to ``tensor``."""
    gradient = np.zeros_like(tensor.data)
    iterator = np.nditer(tensor.data, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = tensor.data[index]

        tensor.data[index] = original + epsilon
        high = float(function().data)
        tensor.data[index] = original - epsilon
        low = float(function().data)
        tensor.data[index] = original

        gradient[index] = (high - low) / (2 * epsilon)
        iterator.iternext()
    return gradient


def assert_gradient_matches(build, tensors, tolerance=TOLERANCE):
    """Backward-pass gradients must match finite differences."""
    for tensor in tensors:
        tensor.zero_grad()
    output = build()
    output.backward()
    analytical = [tensor.grad.copy() for tensor in tensors]

    for tensor, expected in zip(tensors, analytical):
        numerical = numerical_gradient(build, tensor)
        assert np.allclose(expected, numerical, atol=1e-4, rtol=1e-3), (
            f"gradient mismatch\nanalytical:\n{expected}\n"
            f"numerical:\n{numerical}")


@pytest.mark.parametrize("operation", [
    lambda a, b: (a * b).sum_axis(),
    lambda a, b: (a + b).sum_axis(),
    lambda a, b: (a - b).sum_axis(),
    lambda a, b: (a / b).sum_axis(),
    lambda a, b: (a @ b.transpose_last2()).sum_axis(),
])
def test_binary_operation_gradients(operation):
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(3, 4)) + 2.0)   # offset keeps division stable
    assert_gradient_matches(lambda: operation(a, b), [a, b])


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu", "gelu"])
def test_activation_gradients(activation):
    rng = np.random.default_rng(2)
    # Avoid values at exactly zero, where ReLU is not differentiable.
    data = rng.normal(size=(4, 3))
    data[np.abs(data) < 1e-3] = 0.5
    x = Tensor(data)
    assert_gradient_matches(lambda: getattr(x, activation)().sum_axis(), [x])


def test_softmax_and_log_gradients():
    rng = np.random.default_rng(3)
    x = Tensor(rng.normal(size=(5, 4)))
    assert_gradient_matches(lambda: x.softmax().log().sum_axis(), [x])


def test_softmax_cross_entropy_gradient():
    rng = np.random.default_rng(4)
    logits = Tensor(rng.normal(size=(6, 3)))
    labels = np.array([0, 1, 2, 1, 0, 2])
    assert_gradient_matches(
        lambda: softmax_cross_entropy(logits, labels), [logits])


def test_layernorm_gradient():
    rng = np.random.default_rng(5)
    x = Tensor(rng.normal(size=(4, 6)))
    gain = Tensor(np.ones(6))
    bias = Tensor(np.zeros(6))
    assert_gradient_matches(
        lambda: layernorm(x, gain, bias).sum_axis(), [x, gain, bias])


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(6)
    probabilities = Tensor(rng.normal(size=(7, 5)) * 10).softmax().data
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert (probabilities >= 0).all()


def test_softmax_is_numerically_stable_for_large_logits():
    """Shifting by the row max must keep large logits from overflowing."""
    probabilities = Tensor(np.array([[1000.0, 1001.0, 999.0]])).softmax().data
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(), 1.0)


def test_cross_entropy_is_lower_for_confident_correct_predictions():
    confident = softmax_cross_entropy(Tensor([[10.0, 0.0, 0.0]]), np.array([0]))
    unsure = softmax_cross_entropy(Tensor([[0.1, 0.0, 0.0]]), np.array([0]))
    assert float(confident.data) < float(unsure.data)


def test_clip_grad_norm_scales_only_when_over_the_limit():
    small = Tensor(np.zeros((2, 2)))
    small.grad = np.full((2, 2), 0.1)
    before = small.grad.copy()
    clip_grad_norm([small], max_norm=5.0)
    assert np.allclose(small.grad, before), "under the limit must be untouched"

    large = Tensor(np.zeros((2, 2)))
    large.grad = np.full((2, 2), 100.0)
    total = clip_grad_norm([large], max_norm=5.0)
    assert total > 5.0
    assert np.linalg.norm(large.grad) == pytest.approx(5.0, abs=1e-3)


def test_adam_reduces_a_simple_quadratic():
    """The optimiser must actually descend: minimise (x - 3)^2."""
    x = Tensor(np.array([0.0]))
    optimiser = Adam([x], lr=0.1)
    losses = []
    for _ in range(300):
        optimiser.zero_grad()
        loss = ((x - 3.0) * (x - 3.0)).sum_axis()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]
    assert x.data[0] == pytest.approx(3.0, abs=1e-2)


def test_masked_timestep_contributes_no_gradient():
    """The padding mask must genuinely stop padded steps from training.

    This mirrors how the LSTM freezes state past a sequence's real length: a
    zero mask has to zero the gradient, or padded timesteps would silently
    train the model.
    """
    value = Tensor(np.array([[2.0], [3.0]]))
    mask = Tensor(np.array([[1.0], [0.0]]))
    output = (value * mask).sum_axis()
    output.backward()
    assert value.grad[0, 0] == pytest.approx(1.0)
    assert value.grad[1, 0] == pytest.approx(0.0)
