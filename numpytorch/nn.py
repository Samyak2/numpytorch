from typing import Tuple, Iterator, List
import numpy as np
from .activations import Activation, Identity


class Param:
    """Stores a trainable parameter and its gradient

    :param data: value to store
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None


class Module:
    """A general module to define a layer or multiple layers of a neural network"""

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass

        :param x: Input for forward pass
        """
        raise NotImplementedError("This module does not have a forward pass")

    def backward(self, dA: np.ndarray, *args):
        """Backward pass. Calculates and stores gradients.

        :param dA: gradient of the output of this module
        """
        raise NotImplementedError("This module does not have a backward pass")

    @staticmethod
    def _add_param(params: List, attr):
        if isinstance(attr, Param):
            params.append(attr)
        elif isinstance(attr, Module):
            params.extend(attr.parameters())

    def parameters(self) -> Tuple[Param, ...]:
        """Returns all trainable parameters"""
        params = []
        for attr in vars(self).values():
            self._add_param(params, attr)
            # check if attr is iterablel
            try:
                iterator = iter(attr)
                for x in iterator:
                    self._add_param(params, x)
            except TypeError:
                pass
        return tuple(params)


class Dense(Module):
    """A fully connected dense layer

    :param in_len: Size of input
    :type in_len: int
    :param out_len: Size of output
    :type out_len: int
    :param activation: Activation function. Default is identity.
    :param xavier_init: Whether to use Xavier initialisation.
        The weights are multiplied by sqrt(2 / in_len)
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        activation: Activation = Identity,
        xavier_init: bool = False,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len

        self.w = Param(np.random.randn(out_len, in_len))
        if xavier_init:
            self.w.data *= np.sqrt(2 / in_len)
        self.b = Param(np.random.randn(out_len))
        self.activation = activation()

        self.z = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert (
            x.shape[1] == self.in_len
        ), f"Input must be 2D array with size (m, {self.in_len})"
        if len(x.shape) < 2:
            x = x.reshape((1, x.shape[0]))
        # Tested
        self.z = np.einsum("oi,mi->mo", self.w.data, x) + self.b.data
        return self.activation(self.z)

    # pylint: disable=arguments-differ
    def backward(self, dA: np.ndarray, A_prev: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # All tested
        dz = dA * self.activation(self.z, back=True)
        self.w.grad = np.einsum("mo,mi->oi", dz, A_prev) / dz.shape[0]
        self.b.grad = np.einsum("mo->o", dz) / dz.shape[0]
        dA_prev = np.einsum("oi,mo->mi", self.w.data, dz)
        return dA_prev


class Sequential(Module):
    """Sequential layers to stack multiple layers one after another.
    All layers must be passed as different parameters.
    """

    def __init__(self, *modules: Iterator[Module]):
        super().__init__()
        self.modules = tuple(modules)
        self.outs = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outs = [x]
        for module in self.modules:
            a = module(self.outs[-1])
            self.outs.append(a)
        return self.outs[-1]

    # pylint: disable=arguments-differ
    def backward(self, dA: np.ndarray):
        for module, prev_out in zip(reversed(self.modules), reversed(self.outs[:-1])):
            dA = module.backward(dA, prev_out)
