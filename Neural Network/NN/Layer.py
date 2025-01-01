import numpy as np

from Neuron import Neuron
from Activation import Logistic


def backpropagate_layer(neurons, delta_next, w_next, alpha=0.1):
    w_pasts = []
    deltas = []
    
    for i,neuron in enumerate(neurons):
        delta, w_past = neuron.update(delta_next, w_next[i], alpha=alpha)
        w_pasts.append(w_past)
        deltas.append(delta)
        
    return np.array(deltas).T, np.array(w_pasts).T

class Layer(object):
    def __init__(self,size, in_size, activation=Logistic()):
        self.neurons = [Neuron(in_size, activation) for i in range(size)]
        
    def randomize(self):
        for neuron in self.neurons:
            neuron.randomize()
        
    def forward(self, X):
        activations = []
        for neuron in self.neurons:
            activations.append(neuron.activate(X))
        return np.array(activations).T
    
    def backward(self, Delta_next, W_next, alpha=1.):
        return backpropagate_layer(self.neurons, Delta_next, W_next, alpha=alpha)
 