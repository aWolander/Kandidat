import random
from Fully_Connected_Layer import *
from Network import Network


class FCNN(Network):
    def __init__(self, layer_specs):
        self.layer_specs = layer_specs
        super().__init__(self.construct_network())

        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

        self.loss_func = "entropy"
        self.epochs = 4
        self.batch_size = 500
        self.learning_rate = 1
        self.get_test_data = False

    def set_data(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def construct_network(self):
        layers = []
        for layer_type in self.layer_specs:
            temp_weight = np.random.randn(layer_type[1], layer_type[0])/2
            temp_bias = np.random.randn(layer_type[1], 1)/2
            layers.append(Fully_Connected_Layer(temp_weight, temp_bias, layer_type[2]))
        return layers

    def train_network(self):
        if self.train_X is None or self.train_y is None or self.test_y is None or self.test_X is None:
            raise "Data is not assigned"

        combined_training = list(zip(self.train_X, self.train_y))
        if self.get_test_data:
            loss_history = []
            correct_guess_history = []
            (correct_guesses, total_loss) = self.test_network()
            self.progress_printer(total_loss, correct_guesses)
            loss_history.append(total_loss)
            correct_guess_history.append(correct_guesses)
        for epoch in range(self.epochs):
            random.shuffle(combined_training)
            for start_point in range(0, len(combined_training), self.batch_size):
                batch = combined_training[start_point:start_point + self.batch_size]
                self.__backprop_batch(batch)
                # This uses len(batch) because if, for example, batch_size = 400 and combined_training = 1500
                # Then the last batch has size 300, not 400
                self.__gradient_descent(len(batch))
            if self.get_test_data:
                (correct_guesses, total_loss) = self.test_network()
                self.progress_printer(total_loss, correct_guesses)
                loss_history.append(total_loss)
                correct_guess_history.append(correct_guesses)
        if self.get_test_data:
            return [correct_guess_history, loss_history]

    def __backprop_batch(self, batch):
        self.__reset_layer_gradients()
        for (image, y) in batch:
            self.__backprop(image, y)

    def __reset_layer_gradients(self):
        for layer in self.layers:
            layer.reset_gradient()

    def __backprop(self, image, y):
        a_layers, z_layers = self.forwardprop_list(image)
        delta = self.__first_delta(z_layers[-1], a_layers[-1], y)

        # the loop can probably go from 1 and exclude this line, but i cant be bothered. it works.
        # indeces get weird if it's changed
        self.layers[-1].first_layer_backward(delta, a_layers[-2])
        for i in range(2, len(self.layers) + 1):
            # This is ugly, but necessary i believe
            delta = self.layers[-i].backward(self.layers[-i + 1].get_weight(), delta, z_layers[-i], a_layers[-i - 1])

    def __gradient_descent(self, batch_size):
        for layer in self.layers:
            layer.descend(batch_size, self.learning_rate)

    def forwardprop_list(self, image):
        z = image
        a = sigmoid(z)
        a_layers = [a]
        z_layers = [z]
        for layer in self.layers:
            (a, z) = layer.forward(a)
            a_layers.append(a)
            z_layers.append(z)
        return a_layers, z_layers

    def forwardprop(self, image):
        z = image
        a = sigmoid(z)
        for layer in self.layers:
            a, _ = layer.forward(a)
        return a

    def test_network(self):
        total_loss = 0
        correct_guesses = 0
        for (image, y) in zip(self.test_X, self.test_y):
            y_hat = self.forwardprop(image)
            total_loss += loss(y_hat, y, self.loss_func)
            if np.argmax(y_hat) == np.argmax(y):
                correct_guesses += 1
        total_loss = total_loss / len(self.test_X)
        return [correct_guesses, total_loss]

    # could maybe be a part of the Network class, but i want it to *just* be a collection of layers
    def progress_printer(self, total_loss, correct_guesses):
        print("Loss: {0}.\nNumber of correct guesses: {1} out of {2}.".format(str(total_loss),
                                                                              str(correct_guesses),
                                                                              str(len(self.test_X))))

    # kan kanske undvika alla if satser om allt är dot products. borde också inte komma åt
    # indivduella aktiveringsfunktioner direkt.
    def __first_delta(self, z_layer, y_hat, y):
        if self.loss_func == "L2" and self.layers[-1].activation_func == "sigmoid":
            delta = loss_L2_prim(y_hat, y) * sigmoid_prim(z_layer)
            return delta
        if self.loss_func == "L2" and self.layers[-1].activation_func == "softmax":
            return np.dot(softmax_prim(z_layer), loss_L2_prim(y_hat, y))
        if self.loss_func == "entropy" and self.layers[-1].activation_func == "softmax":
            return np.dot(softmax_prim(z_layer), cross_entropy_prim(y_hat, y))
        if self.loss_func == "entropy" and self.layers[-1].activation_func == "sigmoid":
            # dont know if this works, should logically not be used anyways
            return cross_entropy_prim(y_hat, y) * sigmoid_prim(z_layer)
        raise "Combination of loss and acitvation function does not exist"

    def change_layers(self, new_layer_specs):
        self.layer_specs = new_layer_specs
        self.construct_network()

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_get_test_data(self, get_test_data):
        self.get_test_data = get_test_data
