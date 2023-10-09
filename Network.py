import copy
from quantization_function import *

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forwardprop(self, image):
        pass

    def get_copy(self):
        temp_network = copy.deepcopy(self)
        temp_network.set_layers(self.get_layers_copy())
        return temp_network

    def get_layers_copy(self):
        temp_layers = []
        for layer in self.layers:
            temp_layers.append(layer.get_copy())
        return temp_layers

    def get_layers(self):
        return self.layers

    def quantized_network(self, q):
        temp_network = self.get_copy()
        if q == 0:
            return temp_network
        for layer in temp_network.get_layers():
            quantized_weight = quantize(layer.get_weight(), q)
            quantized_bias = quantize(layer.get_bias(), q)
            layer.set_weight(quantized_weight)
            layer.set_bias(quantized_bias)
        return temp_network

    def set_layers(self, other_layers):
        for (layer, other_layer) in zip(self.layers, other_layers):
            layer.set_weight(other_layer.get_weight())
            layer.set_bias(other_layer.get_bias())

    def shape(self):
        output = []
        for layer in self.layers:
            output.append(layer.shape())
        return output

    def __sub__(self, other):
        temp_network = self.get_copy()
        temp_layers = []
        for (own_layer, other_layer) in zip(temp_network.get_layers(), other.get_layers()):
            temp_layers.append(own_layer - other_layer)
        temp_network.set_layers(temp_layers)
        return temp_network

    def __add__(self, other):
        temp_network = self.get_copy()
        temp_layers = []
        for (own_layer, other_layer) in zip(temp_network.get_layers(), other.get_layers()):
            temp_layers.append(own_layer + other_layer)
        temp_network.set_layers(temp_layers)
        return temp_network

    def __mul__(self, other):
        temp_network = self.get_copy()
        temp_layers = []
        for own_layer in temp_network.get_layers():
            temp_layers.append(own_layer * other)
        temp_network.set_layers(temp_layers)
        return temp_network

    def __truediv__(self, other):
        temp_network = self.get_copy()
        temp_layers = []
        for own_layer in temp_network.get_layers():
            temp_layers.append(own_layer / other)
        temp_network.set_layers(temp_layers)
        return temp_network

    def __str__(self):
        temp_layers = []
        for layer in self.layers:
            temp_layers.append(str(layer.get_copy()))
        return str(temp_layers)
