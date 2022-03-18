
"""
Reference
https://github.com/yuchen071/Feedforward-Classification-Network
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

#%% functions
def img_process(img_list):
    out = []
    for i in range(len(img_list)):
        # row major
        out.append(np.ravel(img_list[i]) / 255)
    return np.array(out)

def init_wb(nn_list):
    weight = []
    for _, dims in enumerate(nn_list):
        std = np.sqrt(2.0 / (dims[0] + dims[1]))
        w = randn(dims[1], dims[0]) * std
        weight.append(w)

    bias = []
    for _, dims in enumerate(nn_list):
        # init as zeros
        b = np.zeros((dims[1], 1))
        bias.append(b)

    return weight, bias

def dataloader(images, labels, batch_size):
    num_batch = np.ceil(len(labels) / batch_size).astype(np.int16)
    out = []
    for i in range(num_batch):
        tmp = [
            images[batch_size*i : batch_size*(i+1), :].T,
            labels[batch_size*i : batch_size*(i+1)]
            ]
        out.append(tmp)

    return out

def label2target(label, nn_shape):
    # label vector to target matrix
    target = np.zeros(nn_shape)
    for i in range(nn_shape[1]):
        target[label[i]][i] = 1

    return target

def act_func(x, act):
    if act == "relu":
        y = np.maximum(0, x)
    elif act == "softmax":
        m = x.max(axis=0)
        y = np.exp(x - m)
        y = y/np.sum(y, axis=0)
    return y

def loss_func(x, target):
    epsilon = 1e-15     # prevent log(0)    # epsilon = np.finfo(float).eps
    y = -np.sum(target * np.log(x + epsilon)) / target.shape[1]
    return y

#%% train
def train(train_config, dataset):
    # initialize train and test dataset
    trainloader = dataloader(dataset["train_images"], dataset["train_labels"],train_config["batch_size"])
    testloader = dataloader(dataset["test_images"], dataset["test_labels"],train_config["batch_size"])
    num_layers = len(train_config['nn'])    # input layer not included
    # initialize weights and biases
    weight, bias = init_wb(train_config['nn'])
    
    iteration_train=0
    train_loss = []
    test_loss = []
    
    # train
    for batch_id, data in enumerate(trainloader):
        x, label = data[0], data[1]
        batch_len = len(label)

        a = [] # include input layer
        z = [] # input layer not included

        # forward
        a.append(x)
        for layer_id in range(num_layers):
            z.append(weight[layer_id].dot(a[layer_id])+bias[layer_id])
            a.append(act_func(z[-1], act_list[layer_id]))

        # label vector to target matrix
        target = label2target(label, a[-1].shape)

        # loss & accuracy
        loss_train = loss_func(a[-1], target)
        train_loss.append(loss_train)

        # backward
        # cross entropy has to be paired with softmax to backprop
        dE_dz = []
        dE_dz.append(a[-1]-target)

        # get rest of dE_dz list
        for layer_id in reversed(range(num_layers-1)):
            dE_da = weight[layer_id+1].T.dot(dE_dz[-1])
            dE_dz.append((z[layer_id]>0) * 1 * dE_da)
        dE_dz = dE_dz[::-1] # reverse list

        # get dE_dw, dE_db list
        dE_dw = []
        dE_db = []
        for layer_id in range(num_layers):
            dE_dw.append(dE_dz[layer_id].dot(a[layer_id].T))
            dE_db.append(np.sum(dE_dz[layer_id], 1, keepdims=True))

            # mini-batch SGD
            weight[layer_id] = weight[layer_id] - lr * dE_dw[layer_id] / batch_len
            bias[layer_id] = bias[layer_id] - lr * dE_db[layer_id] / batch_len
            iteration_train+=1 
                
            # test
            if (iteration_train%500==0):
                for batch_id, data in enumerate(testloader):
                    x, label = data[0], data[1]
                    batch_len = len(label)

                    a = [] # include input layer
                    z = [] # input layer not included

                    # forward
                    a.append(x)
                    for layer_id in range(num_layers):
                        z.append(weight[layer_id].dot(a[layer_id])+bias[layer_id])
                        a.append(act_func(z[-1], act_list[layer_id]))

                    # label vector to target matrix
                    target = label2target(label, a[-1].shape)

                    # loss & accuracy
                    loss = loss_func(a[-1], target)
                    test_loss.append(loss)
    # plot
    plt.figure()
    plt.plot(train_loss, label="Training")
    plt.plot(test_loss, label="Testing")
    plt.title("Loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.show()

#%% Main
if __name__ == "__main__":
    # config file
    train_config = dict()
    train_config["batch_size"] = 100
    train_config["nn"] = [[784,2048],[2048,1024],[1024,512],[512,10]]
    act_list = ["relu", "relu", "relu", "softmax"]
    lr = 1e-3

    # read train and test npz
    dataset = dict()
    mnist_data = np.load("mnist.npz")

    dataset["train_images"] = img_process(mnist_data['x_train'])
    dataset["train_labels"] = mnist_data['y_train']
    dataset["test_images"] = img_process(mnist_data['x_test'])
    dataset["test_labels"] = mnist_data['y_test']

    train(train_config, dataset)
