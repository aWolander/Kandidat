import copy
import numpy as np


class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, input_tensor: np.array) -> np.array:
        pass

    def backward(self, *params):
        pass

    def reset_gradient(self) -> None:
        pass

    def shape(self):
        return (np.shape(self.weight), np.shape(self.bias))

    def get_weight(self):
        return self.weight.copy()

    def get_bias(self):
        return self.bias.copy()

    def set_weight(self, new_weight):
        # probably not neccessary and possibly memory inefficient to do .copy().
        # Pointers have caused too many issues for me to change this.
        self.weight = new_weight.copy()

    def set_bias(self, new_bias):
        self.bias = new_bias.copy()

    def get_copy(self):
        return Layer(self.get_weight(), self.get_bias())

    def __str__(self):
        return str([self.get_weight(), self.get_bias()])

    def __sub__(self, other):
        # copy.deepcopy *might* not be neccessary for these
        temp_layer = copy.deepcopy(self)
        temp_layer.set_weight(temp_layer.get_weight() - other.get_weight())
        temp_layer.set_bias(temp_layer.get_bias() - other.get_bias())
        return temp_layer

    def __add__(self, other):
        temp_layer = copy.deepcopy(self)
        temp_layer.set_weight(temp_layer.get_weight() + other.get_weight())
        temp_layer.set_bias(temp_layer.get_bias() + other.get_bias())
        return temp_layer

    def __mul__(self, other):
        temp_layer = copy.deepcopy(self)
        temp_layer.set_weight(temp_layer.get_weight() * other)
        temp_layer.set_bias(temp_layer.get_bias() * other)
        return temp_layer

    def __truediv__(self, other):
        temp_layer = copy.deepcopy(self)
        temp_layer.set_weight(temp_layer.get_weight() / other)
        temp_layer.set_bias(temp_layer.get_bias() / other)
        return temp_layer
