import numpy as np

#standard interface to implement
class Activation(object) :
    def __init__(self):
        pass
    
    def activate(self):
        pass
    
    def partial(self):
        pass
    
#logistic activation
class Logistic(Activation):
    def __init__(self):
        pass
    
    def activate(self,Z):
        return 1 / (1+np.exp(-Z))
    
    def partial(self, Z, A):
        return A * (1-A)
    
#Relu activation
class ReLU(Activation):
    def __init__(self):
        pass
    
    def activate(self, Z):
        return np.array([float(max(i,0)) for i in Z])
    
    def partial(self, Z, H):
        return np.array([float(1) if i > 0 else float(0) for i in H])