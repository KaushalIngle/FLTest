from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import models.utils as utils
lr = 0.01 
comms_round = 10
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )   

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    


def create_model():
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(784, 10)
    return global_model

def iterate_client_model(weight, clients_batched, client,):
    smlp_local = SimpleMLP()
    local_model = smlp_local.build(784, 10)
    local_model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=metrics)
    
    #set local model weight to the weight of the global model
    local_model.set_weights(weight)
    
    #fit local model with client's data
    local_model.fit(clients_batched[client], epochs=1, verbose=0)
    
    #scale the model weights and add to list
    scaling_factor = utils.weight_scalling_factor(clients_batched, client)
    scaled_weights = utils.scale_model_weights(local_model.get_weights(), scaling_factor)
    
    # local_weights = local_model.get_weights()
    #clear session to free memory after each communication round
    K.clear_session()
    return scaled_weights