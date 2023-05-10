"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with pytorch instead of numpy
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader


# This class is implemented for you to hold datasets. It can be used to initialize a data loader or not. You may choose to use it or not.  

class MyDataset(Dataset):
    def __init__(self, x, y, pr_type):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = (torch.long if pr_type == "classification"
                                                     else torch.float32))
    def __len__(self):
        return self.x.size()[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# NOTE: you need to complete two classes and one function below. Please see instructions marked by "TODO".   

class DenseLayer(nn.Module):
    """
    Implement a dense layer 
    """
    def __init__(self, input_dim, output_dim, activation, reg_weight, init='autograder'):

        """
        Initialize weights of the DenseLayer. 
        args:
            input_dim: integer, the dimension of the input layer
            output_dim: integer, the dimension of the output layer
            activation: string, can be 'linear', 'relu', 'tanh', 'sigmoid', or 'softmax'. 
                        It specifies the activation function of the layer
            reg_weight: the regularization weight/strength, the lambda value in a regularization 
                        term 0.5 * \lambda * ||W||_2^2
            param_init: this input is used by autograder. Please do NOT touch it. 
        """


        super(DenseLayer, self).__init__()

        # set initial values for weights and bias terms. 
        param_init = dict(W=None, b=None)

        if init == 'autograder':
            np.random.seed(137)
            param_init['W'] = np.random.random_sample((input_dim, output_dim)) 
            param_init['b'] = np.random.random_sample((output_dim, )) 
        else:
            # TODO: please do your own initialization using the same dimension as above
            # Note: bad initializations may lead to bad performance later
            param_init['W'] = (np.random.random_sample((input_dim, output_dim)) -0.5) * 0.1
            param_init['b'] = np.random.random_sample((output_dim, )) 
        
        self.W = nn.Parameter(torch.tensor(param_init['W'], dtype = torch.float32))
        self.b = nn.Parameter(torch.tensor(param_init['b'], dtype = torch.float32))
        # TODO: record input arguments and initialize other necessary variables
        self.activation = activation
        self.reg_weight = reg_weight
        
    
    def forward(self, inputs):
        """
        This function implement the `forward` function of the dense layer
        """

        # TODO: implement the linear transformation
        outputs = torch.matmul(inputs, self.W) + self.b
        # TODO: implement the activation function
        if self.activation == 'linear':
          pass
        elif self.activation == 'relu':
          outputs = torch.relu(outputs)
        elif self.activation == 'tanh':
          outputs = torch.tanh(outputs)
        elif self.activation == 'sigmoid':
          outputs = torch.sigmoid(outputs)
        elif self.activation == 'softmax':
          outputs = nn.functional.softmax(outputs)
        return outputs
       
    
class Feedforward(nn.Module):

    """
    A feedforward neural network. 
    """

    def __init__(self, input_size, depth, hidden_sizes, output_size, reg_weight, task_type):

        """
        Initialize the model. This way of specifying the model architecture is clumsy, but let's use this straightforward
        programming interface so it is easier to see the structure of the program. Later when you program with torch 
        layers, you will see more precise approaches of specifying network architectures.  

        args:
          input_size: integer, the dimension of the input.
          depth:  integer, the depth of the neural network, or the number of connection layers. 
          hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                        The first number is the number of output units of first connection layer, and so on so forth.
          output_size: integer, the number of classes. In our regression problem, please use 1. 
          reg_weight: float, The weight/strength for the regularization term.
          task_type: string, 'regression' or 'classification'. The task type. 
        """

        super(Feedforward, self).__init__()

        # Add a condition to make the program robust 
        if not (depth - len(hidden_sizes)) == 1:
            raise Exception("The depth (%d) of the network should be 1 larger than `hidden_sizes` (%d)." % (depth, len(hidden_sizes)))

         
        # TODO: install all connection layers except the last one
        layer_input_sizes = [input_size] + hidden_sizes
        layer_output_sizes = hidden_sizes

        self.denselayers = nn.ModuleList([])
        for i in range(depth - 1):
            self.denselayers.append(
              DenseLayer(layer_input_sizes[i], layer_output_sizes[i], activation='tanh', reg_weight=reg_weight))

        # TODO: decide the last layer according to the task type
        if task_type == 'regression':
            output_activation = 'linear'
        elif task_type == 'classification':
            output_activation = 'softmax'
        
        self.denselayers.append(
          DenseLayer(layer_input_sizes[-1], output_size, activation=output_activation, reg_weight=reg_weight))
        

    def forward(self, inputs):
        """
        Implement the forward function of the network. 
        """

        #TODO: apply the network function to the input
        outputs = inputs
        for i, layer in enumerate(self.denselayers):
          outputs = layer(outputs)

        return outputs 
    
    def calculate_reg_term(self):
        """
        Compute a regularization term from all model parameters  

        args:
        """

        #
        # TODO: compute the regularization term from all connection weights 
        # Note: there is a convenient alternatives for L2 norm: using weight decay. here we consider a general approach, which
        # can calculate different types of regularization methods. 
        reg_term = 0
        for layer in self.denselayers: 
          reg_term += torch.sum(torch.square(layer.W))

        return reg_term

def train(x_train, y_train, x_val, y_val, depth, hidden_sizes, reg_weight, num_train_epochs, task_type, batch_size = 50, lr = 0.02):

    """
    Train this neural network using stochastic gradient descent.

    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets 
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets 
      depth: integer, the depth of the neural network 
      hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                    The first number is the number of output units of first connection layer, and so on so forth.

      reg_weight: float, the regularization strength.
      num_train_epochs: the number of training epochs.
      task_type: string, 'regression' or 'classification', the type of the learning task.
    """
    if task_type=="regression":
      output_size = 1
    elif task_type == "classification":
      output_size = np.max(y_train) + 1
        
    # TODO: set up dataloaders for the training set and the test set. Please check the DataLoader class. You will need to specify 
    # your batch sizes there. 
    dataloader = DataLoader(MyDataset(x_train, y_train, pr_type=task_type), 
                           shuffle=True, 
                           batch_size = batch_size)

    val_dataloader = DataLoader(MyDataset(x_val, y_val, pr_type=task_type),
                            shuffle=True,
                            batch_size = batch_size)
    

    # TODO: initialize a model with the Feedforward class 
    model = Feedforward(x_train.shape[1], depth, hidden_sizes, output_size, reg_weight, task_type)

    # TODO: initialize an opimizer. You can check the documentation of torch.optim.SGD, but you can certainly try more advanced
    # optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # TODO: decide the loss for the learning problem
    if task_type == 'regression':
        loss_func = nn.MSELoss()
    elif task_type == 'classification':
        loss_func = nn.CrossEntropyLoss(reduction='sum')

    # TODO: train the model and record the training history after every training epoch
    
    history = {"loss": [],     # each entry of the list should be the average training loss over the epoch
              "val_loss": [],  # each entry of the list should be the validation loss over the validation set after the epoch
              "accuracy": []}  # each entry of the list should be the evaluation of the model (e.g. accuracy or MSE) over the 
                               # validation set after the epoch. 
    
    for _ in range(num_train_epochs):
      running_train_loss = 0
      n_batch = 0

      for i, data in enumerate(dataloader):
        x_batch, y_batch = data
        optimizer.zero_grad()

        y_pred = model(x_batch)

        loss = loss_func(y_pred, y_batch) 
        running_train_loss += loss.item()
        loss += reg_weight * model.calculate_reg_term()
        loss.backward()

        n_batch += 1
        optimizer.step()

      
      history['loss'].append((running_train_loss/n_batch))

      running_val_loss= 0
      accuracy_epoch = 0

      for i, data in enumerate(val_dataloader):
        x_batch, y_batch = data
        y_test_pred = model(x_batch)

        val_loss = loss_func(y_test_pred, y_batch)
        running_val_loss += val_loss.item()

        if task_type == 'classification':
          y_test_pred_class = torch.argmax(y_test_pred, dim=1)
          acc = torch.mean((y_test_pred_class == y_batch).float())
          accuracy_epoch += acc.item()

      history['val_loss'].append((running_val_loss/(i+1)))
      if task_type == 'classification':
        history['accuracy'].append((accuracy_epoch/(i+1))) 
      if task_type == 'regression':
        history['accuracy'].append((running_val_loss/(i+1)))
           
    return model, history


