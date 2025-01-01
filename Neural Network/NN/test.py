import numpy as np
from Activation import Logistic, ReLU
from Neuron import Neuron
from Layer import Layer


# layer 2 (2 neurons, 2 inputs)
c2 = Layer(2, 2)

# We should not directly assign  weights 
# Here, it is just to have the same weights of the output neuron of the lecture's example
c2.neurons[0].b = -0.3
c2.neurons[0].w = np.array([0.5, 0.2])
c2.neurons[1].b = 0.5
c2.neurons[1].w = np.array([0.3, 0.4])

delta3 = np.array([[ 0.14523862, -0.02613822], [ 0.1394202, -0.02531591]]).T
w3 = np.array([[0.3, -0.1],[0.5, -0.3]])

a1 = np.array([[2., -1.], [3., 5.]])
a2 = c2.forward(a1)
print("Activations: " + str(a2))

Deltas2, W_pasts2 = c2.backward(delta3, w3)

print("Deltas: " + str(Deltas2))



