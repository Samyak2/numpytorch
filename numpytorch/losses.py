import numpy as np

EPS = 1e-06


class Loss:
    """Generic class to define a loss function"""

    def __init__(self):
        pass

    def __call__(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.forward(y_real, y_pred)

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates and returns the loss value."""
        raise NotImplementedError(
            "Forward pass of this loss function has not been implemented"
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the output predictions"""
        raise NotImplementedError(
            "Backward pass of this loss function has not been implemented"
        )


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss used for binary classification

    Calculates negative log likelihood i.e., the entropy between the real
    distribution and the predicted distribution.
    """

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(
            y_real * np.log(y_pred + EPS) + (1 - y_real) * np.log(1 - y_pred + EPS)
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(y_real / (y_pred + EPS)) + (1 - y_real) / (1 - y_pred + EPS)
