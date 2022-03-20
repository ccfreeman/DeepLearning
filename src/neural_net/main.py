from typing import Tuple
import numpy as np


class NeuralNetwork:
    """A simple implementation of a neural network. Training examples X should be given in the shape of (number of fields, number of training examples), i.e.
    each column vector represents a training example
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int]):
        self.X, self.Y = X, Y
        self.define_network(X=X, Y=Y, n_neurons_hidden_layers=n_neurons_hidden_layers)

    def initialize_weights(self, shape: Tuple[int], shrinking_factor: float):
        """Initialize a matrix of weights with given dimensions and a shrinking factor given by argument
        """
        return np.random.randn(shape[0], shape[1]) * shrinking_factor

    def define_network(self, X: np.ndarray, Y: np.ndarray, n_neurons_hidden_layers: Tuple[int], shrinking_factor: float=0.01):
        """Define the shape of the network. Weights and biases are stored as lists of matrices. Each element of the list represents the weights|biases at a
        given level of the network
        """
        input_layer_size = X.shape[0]
        output_layer_size = Y.shape[0]

        # Define the weights and biases for the input layer (Number of fields in X by Number of neurons in first hidden layer)
        self.W = [self.initialize_weights(shape=(n_neurons_hidden_layers[0], input_layer_size), shrinking_factor=shrinking_factor)]
        self.b = [np.zeros(shape=(n_neurons_hidden_layers[0], 1))]
        # Define the weights and biases for the hidden layers
        self.W += [
            self.initialize_weights(
                shape=(n_neurons_hidden_layers[i], n_neurons_hidden_layers[i-1]), shrinking_factor=shrinking_factor
            ) for i in range(1, len(n_neurons_hidden_layers))
        ]
        self.b += [np.zeros(shape=(n_neurons_hidden_layers[i], 1)) for i in range(1, len(n_neurons_hidden_layers))]
        # Define the weights and biases for the output layer
        self.W += [self.initialize_weights(shape=(output_layer_size, n_neurons_hidden_layers[-1]), shrinking_factor=shrinking_factor)]
        self.b += [np.zeros(shape=(output_layer_size, 1))]

        self.n_layers = len(self.W)

        # Define the activation and derivatives of activation functions for each layer of the network
        self.activations = [np.tanh for layer in range(self.n_layers - 1)] + [self.sigmoid]
        self.derivatives = [self.tanh_prime for layer in range(self.n_layers - 1)] + [self.sigmoid_prime]

    #####################################################################
    ## Define several functions we need for the backpropagation algorithm
    #####################################################################

    def sigmoid(self, arr: np.ndarray):
        """The sigmoid function normalizes values to [0, 1]
        """
        return 1 / (1 + np.exp(-arr))

    def sigmoid_prime(self, A: np.ndarray, Y: np.ndarray):
        """Derivative of the sigmoid function
        """
        return A - Y

    def tanh_prime(self, W: np.ndarray, dZ: np.ndarray, A: np.ndarray):
        """Derivative of the tanh function
        """
        return np.dot(W.T, dZ) * (1 - A**2)

    ##########################################
    ## Define forward and backward propagation
    ##########################################
        
    def forward_propagation(self, X: np.ndarray):
        """Perform one step of forward propagation through all the layers of the network
        """
        # Initialize the activation units
        Z = [None]
        A = [X]
        
        # Compute Z and the activation A for each layer, except for the final layer
        for i in range(self.n_layers):
            # print(f"Processing layer {i}. Shapes: {self.W[i].shape} and {A[-1].shape}")
            Z_ = np.dot(self.W[i], A[-1]) + self.b[i]
            # print(f"Using activation function {self.activations[i]}")
            A_ = self.activations[i](Z_)
            Z.append(Z_)
            A.append(A_)
        # Return both Z and A
        return {'Z': Z, 'A': A}

    def cross_entropy_cost(self, A: np.ndarray, Y: np.ndarray):
        """Compute the cross entropy cost of a set of weights
        """
        # Cache the number of training samples
        m = Y.shape[1]
        cost = -(np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (~Y).T)) / m
        return float(np.squeeze(cost))

    def backward_propagation(self, Y: np.ndarray, A: Tuple[np.ndarray]):
        """Perform the backward propagation algorithm, iterating through each layer of the network from end to start
        and using the derivatives of the activation functions at each layer to determine how we should update our parameters.
        """
        # Cache the number of training samples
        m = self.X.shape[1]

        dZ, dW, db = [], [], []

        # Step backward through each layer of the network and compute the derivatives
        for i in range(self.n_layers-1, 0-1, -1):
            # print(f"Performing backward propagation on layer {i}")
            if i == self.n_layers-1:
                # The final layer of the network uses the sigmoid function for activation
                dZ_ = self.sigmoid_prime(A=A[i+1], Y=Y)
            else:
                dZ_ = self.tanh_prime(W=self.W[i+1], dZ=dZ[-1], A=A[i+1])
            dW_ = np.dot(dZ_, A[i].T) / m
            db_ = dZ_.sum(axis=1, keepdims=True) / m
            
            dZ.append(dZ_)
            dW.append(dW_)
            db.append(db_)

        # We have inserted the dW and db layers in reverse. Correct this and store the gradients for return
        dW.reverse()
        db.reverse()
        gradients = {'dW': dW, 'db': db}
        return gradients 

    def update_parameters(self, dW: Tuple[np.ndarray], db: Tuple[np.ndarray], learning_rate: float):
        """Update weights and biases at a given step of gradient descent
        """
        for i in range(self.n_layers):
            assert dW[i].shape == self.W[i].shape
            self.W[i] -= dW[i] * learning_rate
            self.b[i] -= db[i] * learning_rate

    def train(self, n_iterations: int, learning_rate: float=1.2, print_cost: bool=False):
        """Run through gradient descent steps for a given numbrer of iterations, updating parameters
        with a given learning rate. 
        """
        for i in range(n_iterations):
            forward_results = self.forward_propagation(X=self.X)
            cost = self.cross_entropy_cost(A=forward_results['A'][-1], Y=self.Y)
            gradients = self.backward_propagation(Y=self.Y, A=forward_results['A'])
            self.update_parameters(**gradients, learning_rate=learning_rate)
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

    def predict(self, X: np.ndarray, threshold: float=0.5):
        """Use the learned weights to predict on a new set of data
        """
        return self.forward_propagation(X=X)['A'][-1] > threshold
