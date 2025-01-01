import numpy as np
from Activation import Logistic, ReLU
from Neuron import Neuron

# New neuron with two inputs
n = Neuron(2)
# ---------------------
# We should not affect the weights directly
# Here, it's just to have the same weights of the output neuron in the example seen in the lecture
# We will reproduce the parameters of neuron 1 hidden layer 1 (layer 2)
n.b = -0.3
n.w = np.array([0.5, 0.2])
# ---------------------

# M X Lp (Here A1 = X; the input)
A1 = np.array([[2., -1.], [3., 5.]])
# M X Ln (Delta of the next layer)
Delta3 = np.array([[ 0.14523862, -0.02613822], [ 0.1394202, -0.02531591]]).T
W3_1 = np.array([0.3, -0.1])


A2_1 = n.activate(A1)
print("z2_1 = " + str(n.z))
print("a2_1 = " + str(A2_1))
# The partial derivative of logistic function does not need Z, so we pass 0 (to not calculte it)
print("partial(a2_1) = " + str(n.activation.partial(0, A2_1)))
print("past b = " + str(n.b))

Delta2, W2_past = n.update(Delta3, W3_1) 

print("past w = " + str(W2_past))
print("delta2 = " + str(Delta2))
print("new b = " + str(n.b))
print("new w = " + str(n.w))



