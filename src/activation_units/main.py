import numpy as np
from abc import ABC, abstractmethod


#################################
## Abstract base class definition
#################################

class ActivationUnit(ABC):
    @abstractmethod
    def forward(self):
        ...
    
    @abstractmethod
    def backward(self):
        ...


############################
## Define activation classes
############################


class Sigmoid(ActivationUnit):

    def forward(self, Z: np.ndarray, **kwargs):
        """The sigmoid function normalizes values to [0, 1]
        """
        return 1 / (1 + np.exp(-Z))

    def backward(self, dA: np.ndarray, A: np.ndarray, Z: np.ndarray, **kwargs):
        """Derivative of the sigmoid function
        """
        dZ = dA * A * (1 - A)
        assert dZ.shape == Z.shape
        return dZ


class Tanh(ActivationUnit):

    def forward(self, Z: np.ndarray, **kwargs):
        """The sigmoid function normalizes values to [0, 1]
        """
        return np.tanh(Z)

    def backward(self, W: np.ndarray, dZ: np.ndarray, dA: np.ndarray, **kwargs):
        """Derivative of the sigmoid function
        """
        # TODO: This is probably not right?
        return np.dot(W.T, dZ) * (1 - dA**2)


class Relu(ActivationUnit):

    def forward(self, Z: np.ndarray, **kwargs):
        """The sigmoid function normalizes values to [0, 1]
        """
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray, **kwargs):
        """Derivative of the sigmoid function
        """
        dZ = np.array(dA, copy=True)
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ
    