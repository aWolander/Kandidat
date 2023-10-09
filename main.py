from matplotlib import pyplot as plt
import matplotlib as mpl
from FCNN import *
import data_loader
from Federated_Learning import *

mpl.use('TkAgg')  # matplotlib will not work without this:
# https://stackoverflow.com/questions/75453995/pandas-plot-vars-argument-must-have-dict-attribute

def main():
    global CIFAR_label_names
    CIFAR_label_names = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    # data_CIFAR10 = data_loader.load_CIFAR10()
    data_MNIST = data_loader.load_MNIST()
    print("Data loaded")

    # input_layer_CIFAR = 32*32*3
    input_layer_MNIST = 784
    layer_types = [[input_layer_MNIST, 64, "sigmoid"],
                   [64, 32, "sigmoid"],
                   [32, 16, "sigmoid"],
                   [16, 10, "softmax"]
                   ]
    simple_layers = [[input_layer_MNIST, 30, "sigmoid"],
                     [30, 10, "softmax"]
                     ]

    loss_func = "entropy"
    device_num = 40
    learning_rate = 0.5
    epochs = 50
    batch_size = 500
    test_every_epoch = True
    tau = 4
    q1 = 2
    q2 = 2

    # net2 = FCNN(layer_types)
    # net2.set_data(data_MNIST[0], data_MNIST[1], data_MNIST[2], data_MNIST[3])
    # net2.set_epochs(50)
    # net2.set_batch_size(500)
    # net2.set_learning_rate(1)
    # net2.set_get_test_data(True)
    # [correct_guesses, loss_history] = net2.train_network()

    # net2.train_network(epochs, batch_size, learning_rate, test_every_epoch)
    #
    #
    # FLnet = Federated_Learning(layer_types, data_MNIST)
    # FLnet.set_epochs(50)
    # FLnet.set_q(0, 0)  # no quantization
    # [correct_guesses, loss_history] = FLnet.train_network()
    #    # FLnet.set_device_num(2)
    # FLnet.reset_models(layer_types, data_MNIST)
    # [correct_guesses2, loss_history2] = FLnet.train_network()
    #
    # FLnet.set_device_num(4)
    # FLnet.reset_models(layer_types, data_MNIST)
    # [correct_guesses4, loss_history4] = FLnet.train_network()
    #
    # FLnet.set_device_num(8)
    # FLnet.reset_models(layer_types, data_MNIST)
    # [correct_guesses8, loss_history8] = FLnet.train_network()
    #
    # FLnet.set_device_num(24)
    # FLnet.reset_models(layer_types, data_MNIST)
    # [correct_guesses16, loss_history16] = FLnet.train_network()
    #
    # FLnet.set_device_num(40)
    # FLnet.reset_models(layer_types, data_MNIST)
    # [correct_guesses32, loss_history32] = FLnet.train_network()
    #
    plt.figure()
    plt.subplot(121)
    # plt.plot(correct_guesses)
    # # plt.plot(correct_guesses2)
    # # plt.plot(correct_guesses4)
    # # plt.plot(correct_guesses8)
    # # plt.plot(correct_guesses16)
    # # plt.plot(correct_guesses32)
    # # plt.legend(["M = 2", "M = 4", "M = 8", "M = 24", "M = 40"])
    # plt.ylabel("Correct evaluations")
    # plt.xlabel("Epochs")
    # plt.title("Number of correct evaluations in the test data (out of 10 000)")
    # plt.axis([0, len(correct_guesses), 0, 10_000])
    # #
    # plt.subplot(122)
    # plt.plot(loss_history)
    # # plt.plot(loss_history2)
    # # plt.plot(loss_history4)
    # # plt.plot(loss_history8)
    # # plt.plot(loss_history16)
    # # plt.plot(loss_history32)
    # # plt.legend(["M = 2", "M = 4", "M = 8", "M = 24", "M = 40"])
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.title("Total loss of the test data with cross entropy loss")
    # # # plt.axis([0, len(correct_guesses), 0, 5])
    # #
    # plt.show()

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(correct_guesses2)
    # plt.ylabel("Correct guesses")
    # plt.xlabel("Epochs")
    # plt.title("Number of correct evaluations in the test data (out of 10 000)")
    # plt.axis([0, len(correct_guesses2), 0, 10_000])
    #
    # plt.subplot(122)
    # plt.plot(loss_history2)
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.title("Total loss of the test data")
    # # plt.axis([0, len(correct_guesses), 0, 5])
    # plt.show()







def show_image(image_vec, label, type):
    if type == "CIFAR10":
        image = np.reshape(image_vec, (32, 32, 3))
        # image = np.transpose(image, (1, 2, 0))
        plt.imshow(image, interpolation="nearest")
        plt.title(CIFAR_label_names[np.argmax(label)])
    if type == "MNIST10":
        image = np.reshape(np.atleast_2d(image_vec), (28, 28))
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.title(np.argmax(label))
    plt.show()


if __name__ == '__main__':
    main()
