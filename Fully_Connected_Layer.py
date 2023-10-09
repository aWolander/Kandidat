from Activation_Loss import *
from Layer import Layer


class Fully_Connected_Layer(Layer):
    def __init__(self, weight, bias, activation_func):
        super().__init__(weight, bias)
        self.this_layer_size = np.shape(weight)[1]
        self.next_layer_size = np.shape(weight)[0]
        self.activation_func = activation_func
        self.grad_W = 0
        self.grad_b = 0

    def get_activation_func(self):
        return self.activation_func

    def forward(self, input_vector):
        z = np.dot(self.weight, input_vector) + self.bias
        a = activation(z, self.activation_func)
        return a, z

    def backward(self, prevWeight, delta, prev_z, prev_a):
        delta = np.dot(prevWeight.transpose(), delta) * activation_prim(prev_z, self.activation_func)
        self.grad_W += np.dot(delta, prev_a.transpose())
        self.grad_b += delta
        return delta

    def reset_gradient(self):
        self.grad_W = np.zeros((self.next_layer_size, self.this_layer_size))
        self.grad_b = np.zeros((self.next_layer_size, 1))

    def first_layer_backward(self, delta, activation):
        self.grad_W += np.dot(delta, activation.transpose())
        self.grad_b += delta

    def descend(self, batch_size, learning_rate):
        self.weight -= learning_rate / batch_size * self.grad_W
        self.bias -= learning_rate / batch_size * self.grad_b
