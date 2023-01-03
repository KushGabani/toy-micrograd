import random
from engine import Value


class Neuron:
    def __init__(self, num_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, num_in, num_out):
        self.neurons = [Neuron(num_in) for _ in range(num_out)]

    def __call__(self, x):
        output = [f(x) for f in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MultiLayerPerceptron:
    def __init__(self, num_in, num_out):
        size = [num_in] + num_out
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(num_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
