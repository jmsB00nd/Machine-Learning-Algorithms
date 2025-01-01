import numpy as np
from Activation import Logistic, ReLU
from Neuron import Neuron
from Layer import Layer
from NN import NN


X = np.array([[2., -1.], [3., 5.]])
Y = np.array([0., 1.])

nn = NN(2) # 2 features
nn.add_layer(2) # add a hidden layer with 2 neurons
nn.add_layer(2) # add a hidden layer with 2 neurons
nn.add_layer(1) # add an output layer with 1 neuron

# We should not affect the weights directly
# Here, it's just to have the same weights of the output neuron in the example seen in the lecture
nn.layers[0].neurons[0].b = -0.3
nn.layers[0].neurons[0].w = np.array([0.5, 0.2])
nn.layers[0].neurons[1].b = 0.5
nn.layers[0].neurons[1].w = np.array([0.3, 0.4])

nn.layers[1].neurons[0].b = -0.3
nn.layers[1].neurons[0].w = np.array([0.3, 0.5])
nn.layers[1].neurons[1].b = -0.2
nn.layers[1].neurons[1].w = np.array([-0.1, -0.3])

nn.layers[2].neurons[0].b = 1.
nn.layers[2].neurons[0].w = np.array([0.7, 0.7])

J = nn._one_iteration(X, Y)

print("cost = " + str(J))
print("w4_1 = " + str(nn.layers[2].neurons[0].w))
print("w3_1 = " + str(nn.layers[1].neurons[0].w))
print("w3_2 = " + str(nn.layers[1].neurons[1].w))
print("w2_1 = " + str(nn.layers[0].neurons[0].w))
print("w2_2 = " + str(nn.layers[0].neurons[1].w))

nn.fit(X, Y, nbr_it=200)
print("Prediction: " + str(nn.predict(X)))