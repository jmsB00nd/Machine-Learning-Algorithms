import numpy as np

#standard interface for the loss
class Loss(object):
    def __init__(self):
        pass
        
    def calculate(self):
        pass
    
    def partial(self):
        pass
    

class BCE(Loss):
    def __init__(self):
        pass
    
    def calculate(self,H,Y):
        return -(Y*np.log(H) + (1-Y)*np.log(1-H))
    
    def partial(self,H, Y):
        return (H - Y) / (H - H**2)