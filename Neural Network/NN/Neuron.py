import numpy as np
from Activation import Logistic

def backpropagate_neuron(W, b, z, a, a_past, delta_next, W_next, act, alpha=.1):
    dz = act.partial(z, a)
    delta = dz * np.dot(delta_next, W_next)
    
    dW = np.dot(a_past.T,delta) / a_past.shape[0]
    db = np.mean(delta)
    
    W = W - alpha * dW
    b = b - alpha * db
    return W, float(b), delta

class Neuron(object):
    def __init__(self, in_feature, act=Logistic()):
        self.b = 0
        self.w = np.array([0.]*in_feature)
        self.activation = act
        
    def randomize(self):
        self.w = np.random.rand(len(self.w))
        self.b = np.random.rand(1)[0]
        
    def aggregate(self,X):
        return np.dot(X, self.w) + self.b
    
    def activate(self,X):
        self.a_past = X
        self.z      = self.aggregate(X)
        self.a      = self.activation.activate(self.z)
        return self.a
    
    def update(self, delta_next, w_next, alpha=.1):
        w_past = self.w.copy()
        self.w, self.b, delta = backpropagate_neuron(self.w, self.b, self.z, self.a, self.a_past, delta_next, w_next, self.activation, alpha)
        return delta, w_past
    