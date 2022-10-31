from imutils import paths
import dataset.utils as utils
import os
import tensorflow as tf
import random
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

    
def load_data_iid(num_clients = 10):
    img_path = os.path.dirname(__file__)+'/mnist/data/trainingSet/trainingSet'
    #get the path list using the path object
    image_paths = list(paths.list_images(img_path))

    #apply our function
    image_list, label_list = utils.load(image_paths, verbose=10000)

    #binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)

    #split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                        label_list, 
                                                        test_size=0.1, 
                                                        random_state=42)

    clients = utils.create_clients(X_train, y_train, num_clients, initial='client')

    #process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = utils.batch_data(data)
    
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    
    return clients_batched, test_batched
    
    
def load_data_non_iid( x=1, num_intraclass_clients=10):
        ''' x: none IID severity, 1 means each client will only have one class of data
            num_intraclass_client: number of sub-client to be created from each none IID class,
            e.g for x=1, we could create 10 further clients by splitting each class into 10
            '''
        img_path = os.path.dirname(__file__)+'/mnist/data/trainingSet/trainingSet'
        #get the path list using the path object
        image_paths = list(paths.list_images(img_path))

        #apply our function
        image_list, label_list = utils.load(image_paths, verbose=10000)

        #binarize the labels
        lb = LabelBinarizer()
        label_list = lb.fit_transform(label_list)

        #split data into training and test set
        X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                            label_list, 
                                                            test_size=0.1, 
                                                            random_state=42)
        
    
        #create unique label list and shuffle
        # unique_labels = np.unique(np.array(y_train))
        new_data = [list(y) for y in set([tuple(x) for x in y_train])]
        random.shuffle(new_data)
        # print(new_data)
        #create sub label lists based on x
        sub_lab_list = [new_data[i:i + x] for i in range(0, len(new_data), x)]
        non_iid_clients_batched = dict()
        # print(sub_lab_list)
        # print(y_train)
        for item in sub_lab_list:
            # print(item)
            # class_data = [(image, label) for (image, label) in zip(X_train, y_train) if label in item]
            class_data = []
            for (image,label) in zip(X_train, y_train):
                if list(label) in item:
                    # print(list(label))
                    # print("x")
                    class_data.append((image,label))
            
            # print(class_data)
            #decouple tuple list into seperate image and label lists
            images, labels = zip(*class_data)
            
            # create formated client initials
            initial = ''
            for lab in item:
                initial = initial + str(lab) + '_'
            
            #create num_intraclass_clients clients from the class 
            intraclass_clients = utils.create_clients(list(images), list(labels), num_intraclass_clients, initial)
            for (client_name, data) in intraclass_clients.items():
                non_iid_clients_batched[client_name] = utils.batch_data(data)
            #append intraclass clients to main clients'dict
            # non_iid_clients_batched.update(intraclass_clients)
        
        test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

        return non_iid_clients_batched, test_batched