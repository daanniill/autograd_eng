# Neural Net
import random
from engine import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # create random weights betweeen -1 and 1 for number of inputted (nin)
        self.b = Value(random.uniform(-1, 1)) # create random bias
                       
    def __call__(self, x): # defines behavior when a neuron object is called
        # w * x + b
        
        # creates a new list where the weights and x inputs are paired ex: [(Value(data=-0.8019805993604605), 2.0), (Value(data=-0.677962308555665), 3.0)]
        act = sum(wi*xi for wi, xi, in zip(self.w, x)) + self.b
        out = act.tanh()
        return out

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] # define a layer of neurons based on the number of inputs coming into this layer and how many neurons we want to output to the next layer (nout)
    
    def __call__(self, x): # defines behavior for when Layer object is called, returns list of neurons in layer
        outs = [n(x) for n in self.neurons] 
        return outs[0] if len(outs) == 1 else outs

class MLP:

    # in this case, nouts is a list of the sizes of each individual layer that we will want in our mlp
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # creates a new list of the full mlp ex: 3 inputs, [4, 4, 1] outputs => MLP: [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] # creates a layer for each element in size list

    # returns the final layer, or output, of the mlp
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x