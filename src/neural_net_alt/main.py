from copy import deepcopy
from typing import Dict, Tuple
import numpy as np


class NeuralNetwork:
    """A simple implementation of a neural network. Training examples X should be given in the shape of (number of fields, number of training examples), i.e.
    each column vector represents a training example.

    To use the NeuralNetwork class, initialize it with an X and Y, as well as the number of neurons you want in each layer of the network. Train the example using
    the ```train``` method, specifying how many iterations to execute. You can retrieve accuracy, or predict on unseen samples by using the ```predict``` method.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int], shrinking_factor: float=.1, random_seed: int=1):
        self.X, self.Y = X, Y
        self.parameters = self.define_network(
            X=X, Y=Y, n_neurons_hidden_layers=n_neurons_hidden_layers, random_seed=random_seed, shrinking_factor=shrinking_factor
        )

    def initialize_weights(self, shape: Tuple[int], shrinking_factor: float):
        """Initialize a matrix of weights with given dimensions and a shrinking factor given by argument
        """
        return np.random.randn(shape[0], shape[1]).astype(np.float32) * shrinking_factor

    def define_network(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int], random_seed: int, shrinking_factor: float):
        """Define the shape of the network. Weights and biases are stored as lists of matrices. Each element of the list represents the weights|biases at a
        given level of the network
        """
        np.random.seed(random_seed)
        input_layer_size = X.shape[0]
        output_layer_size = Y.shape[0]

        layer_dims = [input_layer_size] + list(n_neurons_hidden_layers) + [output_layer_size]

        L = len(layer_dims)

        self.W = np.array([self.initialize_weights(shape=(layer_dims[l], layer_dims[l-1]), shrinking_factor=shrinking_factor) for l in range(1, L)], dtype='object')
        self.b = np.array([np.zeros(shape=(layer_dims[l], 1), dtype=np.float32) for l in range(1, L)], dtype='object')

        self.n_layers = L - 1

        # # Define the activation and derivatives of activation functions for each layer of the network
        # self.activations = [self.relu for layer in range(self.n_layers - 1)] + [self.sigmoid]
        # self.derivatives = [self.relu_prime for layer in range(self.n_layers - 1)] + [self.sigmoid_prime]

    #####################################################################
    ## Define several functions we need for the backpropagation algorithm
    #####################################################################

    def sigmoid(self, Z: np.ndarray):
        """The sigmoid function normalizes values to [0, 1]
        """
        return 1 / (1 + np.exp(-Z))

    def sigmoid_prime(self, dA: np.ndarray, A: np.ndarray, Z: np.ndarray):
        """Derivative of the sigmoid function
        """
        # dZ = np.multiply(np.multiply(A, 1 - A), dA)
        # dA * A * (1 - A)
        dZ = dA * A * (1 - A)
        assert dZ.shape == Z.shape
        return dZ

    def tanh(self, Z: np.ndarray):
        """Wrapper for the hyperbolic tangent function
        """
        return np.tanh(Z)

    def tanh_prime(self, W: np.ndarray, dZ: np.ndarray, dA: np.ndarray):
        """Derivative of the tanh function
        """
        return np.dot(W.T, dZ) * (1 - dA**2)

    def relu(self, Z: np.ndarray):
        """
        """
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        return A
    
    def relu_prime(self, dA: np.ndarray, Z: np.ndarray):
        """
        """
        dZ = np.array(dA, copy=True)
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ

    ##########################################
    ## Define forward and backward propagation
    ##########################################
    
    def linear_forward(self, A: np.ndarray, W: np.ndarray, b: np.ndarray):
        """The linear step for a forward pass
        """
        Z = np.dot(W, A) + b
        return Z

    def linear_activation_forward(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str):
        """Execute the linear step for a forward pass and the activation step
        """
        Z = self.linear_forward(A=A_prev, W=W, b=b)
        if activation == 'sigmoid':
            A = self.sigmoid(Z)
        elif activation == 'relu':
            A = self.relu(Z)
        return Z, A

    def forward_propagation(self, X: np.ndarray):
        """Perform one step of forward propagation through all the layers of the network
        """
        A = X
        Z_, A_ = [], []

        for l in range(self.n_layers - 1):
            A_prev = A
            Z, A = self.linear_activation_forward(A_prev=A_prev, W=self.W[l], b=self.b[l], activation='relu')
            Z_.append(Z)
            A_.append(A)

        ZL, AL = self.linear_activation_forward(
            A_prev=A, W=self.W[-1], b=self.b[-1], activation='sigmoid'
        )
        Z_.append(ZL)
        A_.append(AL)
        return Z_, A_

    def cross_entropy_cost(self, AL: np.ndarray, Y: np.ndarray):
        """Compute the cross entropy cost of a set of weights
        """
        # Cache the number of training samples
        m = Y.shape[1]
        cost = -(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)) / m
        return float(np.squeeze(cost))

    def linear_backward(self, dZ: np.ndarray, A_prev: np.ndarray, W: np.ndarray):
        """Execute the backward pass over the linear step at a given layer
        """
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = dZ.sum(axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        assert dA_prev.shape == A_prev.shape
        return dA_prev, dW, db

    def linear_activation_backward(self, dA: np.ndarray, A: np.ndarray, A_prev: np.ndarray, Z: np.ndarray, W: np.ndarray, activation: str):
        """Execute the backward pass over the activation step and the linear step of a given layer
        """
        if activation == 'relu':
            dZ = self.relu_prime(dA=dA, Z=Z)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_prime(dA=dA, A=A, Z=Z)
        dA_prev, dW, db = self.linear_backward(dZ=dZ, A_prev=A_prev, W=W)
        return dA_prev, dW, db

    def backward_propagation(self, Z: Tuple[np.ndarray], A: Tuple[np.ndarray], Y: np.ndarray):
        """Perform the backward propagation algorithm, iterating through each layer of the network from end to start
        and using the derivatives of the activation functions at each layer to determine how we should update our parameters.
        """
        AL, A_prev = A[-1], A[-2]
        ZL, Z_prev = Z[-1], Z[-2]
        WL, W_prev = self.W[-1], self.W[-2]
        Y = Y.reshape(AL.shape)

        dW, db = [], []

        dAL = - ((Y / AL) - ((1 - Y) / (1 - AL)))

        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA=dAL, A=AL, A_prev=A_prev, Z=ZL, W=WL, activation='sigmoid')
        dW.append(dW_temp)
        db.append(db_temp)

        for l in reversed(range(self.n_layers - 1)):
            if l == 0:
                A_, A_prev = A_prev, self.X
                Z_, Z_prev = Z_prev, None
                W_, W_prev = W_prev, None
            else:
                A_, A_prev = A_prev, A[l-1]
                Z_, Z_prev = Z_prev, Z[l-1]
                W_, W_prev = W_prev, self.W[l-1]

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA=dA_prev_temp, A=A_, A_prev=A_prev, Z=Z_, W=W_, activation='relu')
            dW.append(dW_temp)
            db.append(db_temp)
        dW.reverse()
        db.reverse()
        return dW, db

    def update_parameters(self, dW: Tuple[np.ndarray], db: Tuple[np.ndarray], learning_rate: float):
        """Update weights and biases at a given step of gradient descent
        """
        try:
            assert len(dW) == len(self.W)
        except Exception as ex:
            print(len(dW), len(self.W))
            raise ex
        for l in range(self.n_layers):
            try:
                assert dW[l].shape == self.W[l].shape
                assert db[l].shape == self.b[l].shape
            except Exception as ex:
                print(dW[l].shape, self.W[l].shape)
                raise ex
            self.W[l] -= learning_rate * dW[l]
            self.b[l] -= learning_rate * db[l]

    def train(self, n_iterations: int, learning_rate: float=1.2, print_cost: bool=False):
        """Run through gradient descent steps for a given numbrer of iterations, updating parameters
        with a given learning rate. 
        """
        for i in range(n_iterations):
            Z, A = self.forward_propagation(X=self.X)
            cost = self.cross_entropy_cost(AL=A[-1], Y=self.Y)
            dW, db = self.backward_propagation(Z=Z, A=A, Y=self.Y)
            self.update_parameters(dW=dW, db=db, learning_rate=learning_rate)
            # Print the cost every 1000 iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

    def predict(self, X: np.ndarray, threshold: float=0.5):
        """Use the learned weights to predict on a new set of data
        """
        return self.forward_propagation(X=X)[1][-1] > threshold
