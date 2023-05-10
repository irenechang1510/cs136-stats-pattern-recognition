import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
This is a short tutorial of pytorch. After this tutorial, you should know the following concepts:
1. tensors
2. operations
3. variables 
4. gradient calculation 
5. optimizer 

Functions to conisder: torch.sum, torch.matmul, torch.square, torch.randn, torch.tensor
"""
class LRegDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)


def regression_func(x, w, b):
    """
    The function of a linear regression model
    args: 
        x: torch.Tensor with shape (n, d) 
        w: torch.Tensor with shape (d,) 
        b: torch.Tensor with shape ()

    return: 
        y_hat: torch.Tensor with shape [n,]. y_hat = x * w + b (matrix multiplication)
    """
    
    y_hat = torch.matmul(x, w) + b

    return y_hat



def loss_func(y, y_hat):
    """
    The loss function for linear regression

    args:
        y: torch.Tensor with shape (n,) 
        y_hat: torch.Tensor with shape (n,) 

    return:
        loss: torch.Tensor with shape (). loss = (y -  y_hat)^\top (y -  y_hat) 

    """

    # TODO: implement the function. 
    loss = torch.matmul((y - y_hat).t(), y - y_hat)
    
    return loss


def train_lr(x, y, lamb, batch_size = 100):
    """
    Train a linear regression model.

    args:
        x: numpy array with shape (n, d)
        y: numpy array with shape (n, )
        lamb: weight for the regularization term
    """
    N, D = x.shape
    
    # TODO: uncomment the following three line to fetch a batch of instances through data loader
    dataloader = DataLoader(LRegDataset(x, y), 
                           shuffle=True, 
                           batch_size = batch_size)
    
    # TODO: initialize your parameters (w, b) and hyperparameter lamb 
    # You need to set "requires_grad = True" for parameters so they can be optimized
    w = torch.rand(D, dtype=torch.float32, requires_grad = True)
    b = torch.rand(D, dtype=torch.float32, requires_grad = True)
    lamb = torch.tensor(lamb, dtype=torch.float32)

    model = [w, b]

    # TODO: uncomment the following line to get an optimizer
    opt = torch.optim.SGD(model, lr = 1e-2, momentum = 0.99)
    

    for it in range(1, 1001):
        running_loss = 0

        # enumerate through all data batches
        for i, data in enumerate(dataloader):

            x_batch, y_batch = data

            # TODO: for debugging purpose, check the content of a batch. 
            # comment out this breakpoint once you are sure the code is correct

            # print(x_batch)
            # print(y_batch)
            # raise Exception("Debugging -- do not disturb.")

            # TODO: uncomment this line to zero out gradients of your variable 
            # NOTE: it's VERY hard to debug if you forget this line -- always put it in your 
            #       optimization loop 
            opt.zero_grad()

            # TODO: use your model function `regression_func` to compute the prediction y_hat
            y_hat = regression_func(x_batch, w, b)
            
            # TODO: use your loss function `loss_func` to compute the loss. Please remember to add 
            #       the regularization term 
            loss = loss_func(y_batch, y_hat) + lamb * torch.matmul(w, w) 

            # TODO: uncomment this line to backpropagate gradients 
            # Question 1: how does torch track the computation of `loss` and update parameters? 
            # Question 2: how does torch compute gradients of parameters? 
            loss.backward()

            # keep track of the loss over the entire dataset
            running_loss += loss

            # TODO: uncomment this line to update parameters 
            # Question 3: how does torch update parameters? 
            opt.step()
            
        if it % 100 == 1:
            print('loss beocmes ', running_loss.detach().numpy(), ' after ', it, ' iterations.')

    print('loss becomes ', running_loss.detach().numpy(), ' after ', it, ' iterations.')


    return w, b

