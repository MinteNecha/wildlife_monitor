import numpy as np


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = _wrap(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Tensor(-self.data, (self,))

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-_wrap(other))

    def __rsub__(self, other):
        return _wrap(other) + (-self)

    def __mul__(self, other):
        other = _wrap(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _wrap(other)
        out = Tensor(self.data / other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = _wrap(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape)
            other.grad += _unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def transpose_last2(self):
        out = Tensor(np.swapaxes(self.data, -1, -2), (self,))

        def _backward():
            self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))

        def _backward():
            self.grad += out.grad * s * (1 - s)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self.grad += out.grad * (1 - t ** 2)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def gelu(self):
        c = np.sqrt(2.0 / np.pi)
        a = 0.044715
        x = self.data
        inner = c * (x + a * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1 + t), (self,))

        def _backward():
            dinner_dx = c * (1 + 3 * a * x ** 2)
            dt_dx = (1 - t ** 2) * dinner_dx
            dgelu_dx = 0.5 * (1 + t) + 0.5 * x * dt_dx
            self.grad += out.grad * dgelu_dx

        out._backward = _backward
        return out

    def sqrt(self):
        s = np.sqrt(self.data)
        out = Tensor(s, (self,))

        def _backward():
            self.grad += out.grad / (2 * s)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,))

        def _backward():
            self.grad += out.grad / (self.data + 1e-12)

        out._backward = _backward
        return out

    def softmax(self):
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        s = exp / np.sum(exp, axis=-1, keepdims=True)
        out = Tensor(s, (self,))

        def _backward():
            dot = np.sum(out.grad * s, axis=-1, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def sum_axis(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad += g * np.ones_like(self.data)

        out._backward = _backward
        return out

    def mean(self):
        return self.sum_axis(axis=None) / self.data.size

    @staticmethod
    def stack(tensors, axis=1):
        data = np.stack([t.data for t in tensors], axis=axis)
        out = Tensor(data, tuple(tensors))

        def _backward():
            grads = np.split(out.grad, len(tensors), axis=axis)
            for t, g in zip(tensors, grads):
                t.grad += np.squeeze(g, axis=axis)

        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors, axis=-1):
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, tuple(tensors))
        sizes = [t.data.shape[axis] for t in tensors]

        def _backward():
            grads = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
            for t, g in zip(tensors, grads):
                t.grad += g

        out._backward = _backward
        return out
    
    def dropout(self, p=0.1, training=True):
        if not training or p == 0:
            return self
        mask = (np.random.rand(*self.data.shape) > p).astype(np.float64) / (1 - p)
        out = Tensor(self.data * mask, (self,))

        def _backward():
            self.grad += out.grad * mask

        out._backward = _backward
        return out


def layernorm(x, gain, bias, eps=1e-5):
    d = x.data.shape[-1]
    mu = x.sum_axis(axis=-1, keepdims=True) / d
    diff = x - mu
    var = (diff * diff).sum_axis(axis=-1, keepdims=True) / d
    std = (var + eps).sqrt()
    normed = diff / std
    return normed * gain + bias


def softmax_cross_entropy(logits, labels):
    num_classes = logits.data.shape[-1]
    one_hot = Tensor(np.eye(num_classes)[labels])
    probs = logits.softmax()
    selected = (probs * one_hot).sum_axis(axis=-1)
    return (-selected.log()).mean()


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def clip_grad_norm(params, max_norm=5.0):
    total_norm = 0.0
    for p in params:
        total_norm += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale
    return total_norm