from copy import deepcopy
from typing import Dict, Tuple
import numpy as np


class NeuralNetwork:
    """A simple implementation of a neural network. Training examples X should be given in the shape of (number of fields, number of training examples), i.e.
    each column vector represents a training example.

    To use the NeuralNetwork class, initialize it with an X and Y, as well as the number of neurons you want in each layer of the network. Train the example using
    the ```train``` method, specifying how many iterations to execute. You can retrieve accuracy, or predict on unseen samples by using the ```predict``` method.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int], random_seed: int=1):
        self.X, self.Y = X, Y
        self.parameters = self.define_network(X=X, Y=Y, n_neurons_hidden_layers=n_neurons_hidden_layers, random_seed=random_seed)

    def initialize_weights(self, shape: Tuple[int], shrinking_factor: float):
        """Initialize a matrix of weights with given dimensions and a shrinking factor given by argument
        """
        return np.random.randn(shape[0], shape[1]) * shrinking_factor

    def define_network(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int], random_seed: int, shrinking_factor: float=0.01):
        """Define the shape of the network. Weights and biases are stored as lists of matrices. Each element of the list represents the weights|biases at a
        given level of the network
        """
        np.random.seed(random_seed)
        input_layer_size = X.shape[0]
        output_layer_size = Y.shape[0]

        parameters = {}

        layer_sizes = [input_layer_size] + list(n_neurons_hidden_layers) + [output_layer_size]

        L = len(layer_sizes)

        for l in range(1, L):
            parameters['W{}'.format(str(l))] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) / np.sqrt(layer_sizes[l-1])
            parameters['b{}'.format(str(l))] = np.zeros(shape=(layer_sizes[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_sizes[l], layer_sizes[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_sizes[l], 1))

        self.n_layers = len(parameters) // 2

        return parameters

        # self.W, self.b = [None], [None]

        # # Define the weights and biases for the input layer (Number of fields in X by Number of neurons in first hidden layer)
        # self.W += [self.initialize_weights(shape=(n_neurons_hidden_layers[0], input_layer_size), shrinking_factor=shrinking_factor)]
        # self.b += [np.zeros(shape=(n_neurons_hidden_layers[0], 1))]
        # # Define the weights and biases for the hidden layers
        # self.W += [
        #     self.initialize_weights(
        #         shape=(n_neurons_hidden_layers[i], n_neurons_hidden_layers[i-1]), shrinking_factor=shrinking_factor
        #     ) for i in range(1, len(n_neurons_hidden_layers))
        # ]
        # self.b += [np.zeros(shape=(n_neurons_hidden_layers[i], 1)) for i in range(1, len(n_neurons_hidden_layers))]
        # # Define the weights and biases for the output layer
        # self.W += [self.initialize_weights(shape=(output_layer_size, n_neurons_hidden_layers[-1]), shrinking_factor=shrinking_factor)]
        # self.b += [np.zeros(shape=(output_layer_size, 1))]

        # self.n_layers = len(self.W) - 1

        # # Define the activation and derivatives of activation functions for each layer of the network
        # self.activations = [self.relu for layer in range(self.n_layers - 1)] + [self.sigmoid]
        # self.derivatives = [self.relu_prime for layer in range(self.n_layers - 1)] + [self.sigmoid_prime]

    #####################################################################
    ## Define several functions we need for the backpropagation algorithm
    #####################################################################

    def sigmoid(self, Z: np.ndarray):
        """The sigmoid function normalizes values to [0, 1]
        """
        cache = Z
        return 1 / (1 + np.exp(-Z)), cache

    def sigmoid_prime(self, dA: np.ndarray, cache: np.ndarray):
        """Derivative of the sigmoid function
        """
        # dZ = np.multiply(np.multiply(A, 1 - A), dA)
        # dA * A * (1 - A)
        Z = cache

        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
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
        cache = Z
        return A, cache
    
    def relu_prime(self, dA: np.ndarray, cache: np.ndarray):
        """
        """
        Z = cache
        dZ = np.array(dA, copy=True)
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ

    ##########################################
    ## Define forward and backward propagation
    ##########################################
    
    def linear_forward(self, A: np.ndarray, W: np.ndarray, b: np.ndarray):
        """
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str):
        """
        """
        Z, linear_cache = self.linear_forward(A=A_prev, W=W, b=b)
        if activation == 'sigmoid':
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            A, activation_cache = self.relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X: np.ndarray, parameters: np.ndarray):
        """Perform one step of forward propagation through all the layers of the network
        """
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev=A_prev, W=parameters['W{}'.format(l)], b=parameters['b{}'.format(l)], activation='relu'
            )
            caches.append(cache)

        AL, cache = self.linear_activation_forward(
            A_prev=A, W=parameters['W{}'.format(L)], b=parameters['b{}'.format(L)], activation='sigmoid'
        )
        caches.append(cache)
        return AL, caches
        # # Initialize the activation units
        # Z = [None]
        # A = [X]

        # A_ = X
                
        # # Compute Z and the activation A for each layer, except for the final layer
        # for i in range(self.n_layers):
        #     # print(f"Processing layer {i}. Shapes: {self.W[i].shape} and {A[-1].shape}")
        #     Z_ = np.dot(self.W[i], A_) + self.b[i]
        #     # print(f"Using activation function {self.activations[i]}")
        #     if i == self.n_layers-1:
        #         # print("forward pass using sigmoid")
        #         A_ = self.sigmoid(Z=Z_)
        #     else:
        #         # print("forward pass using relu")
        #         A_ = self.relu(Z=Z_)
        #     # A_ = self.activations[i](Z_)
        #     Z.append(Z_)
        #     A.append(A_)
        # # Return both Z and A
        # return {'Z': Z, 'A': A}

    def cross_entropy_cost(self, AL: np.ndarray, Y: np.ndarray):
        """Compute the cross entropy cost of a set of weights
        """
        # Cache the number of training samples
        m = Y.shape[1]
        cost = -(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)) / m
        return float(np.squeeze(cost))

    def linear_backward(self, dZ: np.ndarray, cache: Tuple[np.ndarray]):
        """
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = dZ.sum(axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        assert dA_prev.shape == A_prev.shape
        return dA_prev, dW, db

    def linear_activation_backward(self, dA: np.ndarray, cache: Tuple[Tuple[np.ndarray]], activation: str):
        """
        """
        linear_cache, activation_cache = cache
        if activation == 'relu':
            dZ = self.relu_prime(dA=dA, cache=activation_cache)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_prime(dA=dA, cache=activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ=dZ, cache=linear_cache)
        return dA_prev, dW, db


    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray, caches: Tuple[Tuple[np.ndarray]]):
        """Perform the backward propagation algorithm, iterating through each layer of the network from end to start
        and using the derivatives of the activation functions at each layer to determine how we should update our parameters.
        """
        grads = {}
        L = len(self.parameters) // 2
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - ((Y / AL) - ((1 - Y) / (1 - AL)))

        current_cache = caches[-1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA=dAL, cache=current_cache, activation='sigmoid')
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            assert len(current_cache) == 2
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA=dA_prev_temp, cache=current_cache, activation='relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp
        return grads

        # # Cache the number of training samples
        # m = self.X.shape[1]

        # A_L = A[self.n_layers]
        # Z_L = Z[self.n_layers]
        # W_L = self.W[self.n_layers-1]

        # A_prev = A[self.n_layers-1]

        # Y = Y.reshape(A_L.shape)

        # dZ, dW, db, dA = [], [], [], []

        # # Handle the final layer of the network, which uses a sigmoid function for the activation
        # dA_L = -(np.divide(Y, A_L) - np.divide(1 - Y, 1 - A_L))
        # assert dA_L.shape == A_L.shape
        # dZ_ = self.sigmoid_prime(dA=dA_L, Z=Z_L, A=A_L)

        # # Implementation of the linear portion of backward propagation for the layer   
        # dW_ = np.dot(dZ_, A_prev.T) / m
        # db_ = dZ_.sum(axis=1, keepdims=True) / m
        # dA_ = np.dot(W_L.T, dZ_)
        # assert dA_.shape == A_prev.shape
        
        # dZ.append(dZ_)
        # dW.append(dW_)
        # db.append(db_)
        # dA.append(dA_)

        # # Step backward through each layer of the network and compute the derivatives
        # for i in reversed(range(self.n_layers - 1)):

        #     A_prev = A[i]

        #     dZ_ = self.relu_prime(dA=dA_, Z=Z[i+1])
        #     # dZ_ = self.tanh_prime(W=self.W[i], dZ=dZ[-1], dA=dA[-1])

        #     # Implementation of the linear portion of backward propagation for the layer   
        #     dW_ = np.dot(dZ_, A_prev.T) / m
        #     db_ = dZ_.sum(axis=1, keepdims=True) / m
        #     dA_ = np.dot(self.W[i].T, dZ_)
        #     assert dA_.shape == A_prev.shape
            
        #     dZ.append(dZ_)
        #     dW.append(dW_)
        #     db.append(db_)
        #     dA.append(dA_)

        # # We have inserted the dW and db layers in reverse. Correct this and store the gradients for return
        # dW.reverse()
        # db.reverse()
        # dA.reverse()
        # gradients = {'dW': dW, 'db': db, 'dA': dA}
        # return gradients 

    def update_parameters(self, parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], learning_rate: float):
        """Update weights and biases at a given step of gradient descent
        """
        params = deepcopy(parameters)
        L = len(parameters) // 2

        for l in range(L):
            try:
                assert grads['dW{}'.format(l+1)].shape == params['W{}'.format(l+1)].shape
            except Exception as e:
                print(grads['dW{}'.format(l+1)].shape)
                print(params['W{}'.format(l+1)])
                raise e
            params['W{}'.format(l+1)] -= grads['dW{}'.format(l+1)] * learning_rate
            params['b{}'.format(l+1)] -= grads['db{}'.format(l+1)] * learning_rate
        return params

    def train(self, n_iterations: int, learning_rate: float=1.2, print_cost: bool=False):
        """Run through gradient descent steps for a given numbrer of iterations, updating parameters
        with a given learning rate. 
        """
        parameters = deepcopy(self.parameters)
        for i in range(n_iterations):
            # forward_results = self.forward_propagation(X=self.X)
            AL, caches = self.forward_propagation(X=self.X, parameters=parameters)
            cost = self.cross_entropy_cost(AL=AL, Y=self.Y)
            gradients = self.backward_propagation(AL=AL, Y=self.Y, caches=caches)
            parameters = self.update_parameters(parameters=parameters, grads=gradients, learning_rate=learning_rate)
            # Print the cost every 1000 iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
        self.parameters = parameters

    def predict(self, X: np.ndarray, threshold: float=0.5):
        """Use the learned weights to predict on a new set of data
        """
        return self.forward_propagation(X=X, parameters=self.parameters)[0] > threshold
