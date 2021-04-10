import numpy as np


class Activation:
    """Generic class to define an activation function.

    Usage:

    ```
    a = Activation()
    y = a(x)
    y_grad = a(x, back=True)
    ```
    """

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function

        :param x: inputs
        """
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass of the activation function

        :param x: inputs
        :returns: Gradient of the inputs
        """
        raise NotImplementedError("Backward pass not implemented")

    def __call__(self, x: np.ndarray, back: bool = False) -> np.ndarray:
        if back:
            return self.backward(x)
        return self.forward(x)


class Identity(Activation):
    """Activation function that returns the input

    y = x
    """

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    """Rectified Linear Unit activation function

    y = 0 for x < 0, x for x >= 0
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[z < 0] = 0
        return z

    def backward(self, x: np.ndarray) -> np.ndarray:
        x_grad = x.copy()
        x_grad[x_grad < 0] = 0
        x_grad[x_grad > 0] = 1
        return x_grad


class Sigmoid(Activation):
    """Sigmoid function, specifically the logistic function

    y = 1 / (1 + e^-x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        act = self(x)
        return act * (1 - act)


class LeakyReLU(Activation):
    """Variation of ReLU that has a small constant slope on the negative side

    y = alpha*x for x < 0, x for x >= 0

    :param alpha: The negative slope parameter
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[z < 0] *= self.alpha
        return z

    def backward(self, x: np.ndarray) -> np.ndarray:
        x_g = x.copy()
        x_g[x_g <= 0] = self.alpha
        x_g[x_g > 0] = 1
        return x_g
