from Layer import Layer, backpropagate_layer
from Loss import BCE
from Activation import Logistic
import numpy as np



def backpropagate_network(layers, cost, H, Y):
    J = np.mean(cost.calculate(H, Y))

    delta_past = cost.partial(H, Y)
    w_past = np.ones((delta_past.shape[0], 1)) 
    for layer in reversed(layers):
        delta_past, w_past = backpropagate_layer(layer.neurons, delta_past, w_past)

    return J

class NN(object):
    def __init__(self, in_size, cost=BCE(), alpha=.1):
        self.current_size = in_size 
        self.cost = cost
        self.alpha = alpha
        self.layers = []
        
    def add_layer(self, size, activation=Logistic()):
        new_layer = Layer(size, self.current_size, activation=activation)
        self.layers.append(new_layer)
        self.current_size = size
        
    def randomize(self):
        for layer in self.layers:
            layer.randomize()
    
    def predict(self, X): 
        Y = X
        if self.norm:
            Y = np.where(self.std==0, X, (X - self.mean)/self.std)
            
        for layer in self.layers:
            Y = layer.forward(Y)
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        return np.where(Y < 0.5, 0, 1)
    
    
    def _one_iteration(self, X, Y):
        # forward propagation
        a = X
        for layer in self.layers:
            a = layer.forward(a)
            
        # cost and its derivative calculation
        YY = np.array(Y)
        if YY.ndim < 2 : 
            YY = YY[:, np.newaxis]
        
        # backward propagation 
        J = backpropagate_network(self.layers, self.cost, a, YY)
    
        return J
    
    def fit(self, X, Y, nbr_it=100, norm=False):
        costs = []
        X_norm = X
        self.norm = norm
        if norm:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_norm = np.where(self.std==0, X, (X - self.mean)/self.std)

        for i in range(nbr_it): 
            J = self._one_iteration(X_norm, Y)
            costs.append(J)
        return costs