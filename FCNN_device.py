from FCNN import FCNN
import copy

class FCNN_device:
    def __init__(self, layer_specs, train_X, train_y, test_X, test_y):
        self.model = FCNN(layer_specs)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.model.set_data(train_X, train_y, test_X, test_y)

        self.tau = None
        self.main_model_copy = None
        self.global_model_update = None
        self.local_model_update = False
        self.delta = None
        self.delta_prev = 0
        self.batch_size = None
        self.learning_rate = None
        self.q = None
        self.testval = 0

    def set_run_options(self, loss_func, tau, batch_size, learning_rate, q):
        self.model.set_batch_size(batch_size)
        self.model.set_learning_rate(learning_rate)
        self.model.set_epochs(tau)
        self.model.set_loss_func(loss_func)

        self.q = q
        self.tau = tau

    def recieve_global_model(self, model_update):
        self.global_model_update = copy.deepcopy(model_update)


    def set_layers(self, other_layers):
        self.model.set_layers(other_layers)

    def thread_process(self):

        if self.main_model_copy is None:
            self.main_model_copy = self.global_model_update
        else:
            self.main_model_copy = self.main_model_copy + self.global_model_update

        self.model.set_layers(self.main_model_copy.get_layers())
        # self.model.set_data(self.train_X, self.train_y, self.test_X, self.test_y)  # bad, ugly temporary solution
        self.model.train_network()
        delta_theta = self.model - self.main_model_copy

        if self.delta is None:
            self.local_model_update = delta_theta.quantized_network(self.q)
            self.delta = delta_theta - self.local_model_update
        else:
            self.local_model_update = (delta_theta + self.delta).quantized_network(self.q)
            self.delta = delta_theta + self.delta - self.local_model_update

        return self.local_model_update


    # def transmit_update(self):
    #     return self.local_model_update
