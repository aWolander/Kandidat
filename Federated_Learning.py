from FCNN import FCNN
from FCNN_device import FCNN_device

class Federated_Learning:
    def __init__(self, layer_specs, data):
        self.main_model = None
        self.quantized_main_model = None

        self.device_num = 40
        self.data = data
        self.models = []
        self.q1 = 2
        self.q2 = 2
        self.epochs = 50
        self.batch_size = 500
        self.tau = 4
        self.learning_rate = 1
        self.loss_func = "entropy"
        self.test_network = True

        self.construct_models(layer_specs, data)



    def construct_models(self, layer_specs, data):
        training_data_length = len(data[0])
        self.main_model = FCNN(layer_specs).quantized_network(self.q1)
        self.main_model.set_data(data[0], data[1], data[2], data[3])
        for i in range(self.device_num):
            local_training_data = data[0][training_data_length * i // self.device_num:
                                          training_data_length * (i + 1) // self.device_num]
            local_training_data_labels = data[1][training_data_length * i // self.device_num:
                                          training_data_length * (i + 1) // self.device_num]

            self.models.append(FCNN_device(layer_specs,
                                           local_training_data, local_training_data_labels, data[2], data[3]))

    def reset_models(self, layer_specs, data):
        self.models = []
        self.quantized_main_model = None
        self.construct_models(layer_specs, data)


    def train_network(self):
        if self.test_network:
            correct_guess_history = []
            loss_history = []


        for model in self.models:
            model.set_run_options(self.loss_func, self.tau, self.batch_size, self.learning_rate, self.q2)

        for t in range(self.epochs):
            if self.test_network:
                (correct_guesses, total_loss) = self.main_model.test_network()
                print("FL")
                self.main_model.progress_printer(total_loss, correct_guesses)
                loss_history.append(total_loss)
                correct_guess_history.append(correct_guesses)

            self.global_model_broadcasting()
            local_updates = self.local_update_aggregation()
            self.update_main_model(local_updates)
        if self.test_network:
            return [correct_guess_history, loss_history]


    def global_model_broadcasting(self):
        if self.quantized_main_model is None:
            broadcast = self.main_model.quantized_network(self.q1)
            self.quantized_main_model = broadcast
        else:
            broadcast = (self.main_model - self.quantized_main_model).quantized_network(self.q1)
            self.quantized_main_model = self.quantized_main_model + broadcast

        for model in self.models:
            model.recieve_global_model(broadcast)

    def local_update_aggregation(self):
        local_updates = []
        for model in self.models:
            local_updates.append(model.thread_process())
        return local_updates


    def update_main_model(self, local_model_updates):
        local_update_sum = local_model_updates[0] * len(self.data[0])/self.device_num
        for model_update in local_model_updates[1:]:
            local_update_sum += model_update * len(self.data[0])/self.device_num
        local_update_sum = local_update_sum / len(self.data[0])
        self.main_model = self.quantized_main_model + local_update_sum

    def forwardprop(self, image):
        return self.main_model.forwardprop(image)

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_tau(self, tau):
        self.tau = tau

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_test_network(self, test_network):
        self.test_network = test_network

    def set_q(self, q1, q2):
        self.q1, self.q2 = q1, q2

    def set_device_num(self, device_num):
        self.device_num = device_num