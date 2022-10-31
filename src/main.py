import dataset.mnist as mnist
import models.model as model

import random
import utils.utils as utils
import federated.utils as fedutils
import graph.plot as plot
def test_100():
    clients_batched, test_batched = mnist.load_data_iid(10)

    comms_round = 10

    global_model = model.create_model()
    global_accuracy_list = []
    global_loss_list = []
    # print("x")
    for comm_round in range(comms_round):
        global_weights = global_model.get_weights()
        # print("y")
        scaled_local_weight_list = list()
        
        client_names= list(clients_batched.keys())
        # print("z")
        random.shuffle(client_names)
        for client in client_names:
            # print("A")
            scaled_weights = model.iterate_client_model(global_weights, clients_batched, client)
            scaled_local_weight_list.append(scaled_weights)

        average_weights = fedutils.sum_scaled_weights(scaled_local_weight_list)
        global_model.set_weights(average_weights)
        print("t")
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss = utils.test_model(X_test, Y_test, global_model, comm_round)
            global_accuracy_list.append(global_acc)
            global_loss_list.append(global_loss)
    # plot.graph_results("accuracy", global_accuracy_list)
    # plot.graph_results("loss", global_loss_list)
    

def test_non_iid_100():
    clients_batched, test_batched = mnist.load_data_non_iid(1,10)

    comms_round = 100

    global_model = model.create_model()
    # print("x")
    global_accuracy_list = []
    global_loss_list = []
    for comm_round in range(comms_round):
        global_weights = global_model.get_weights()
        # print("y")
        scaled_local_weight_list = list()
        
        client_names= list(clients_batched.keys())
        # print("z")
        random.shuffle(client_names)
        for client in client_names:
            # print("A")
            scaled_weights = model.iterate_client_model(global_weights, clients_batched, client)
            scaled_local_weight_list.append(scaled_weights)

        average_weights = fedutils.sum_scaled_weights(scaled_local_weight_list)
        global_model.set_weights(average_weights)
        print("t")
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss = utils.test_model(X_test, Y_test, global_model, comm_round)
            global_accuracy_list.append(global_acc)
            global_loss_list.append(global_loss)
    # plot.graph_results("accuracy", global_accuracy_list)
    # plot.graph_results("loss", global_loss_list)

# test_100()
test_non_iid_100()