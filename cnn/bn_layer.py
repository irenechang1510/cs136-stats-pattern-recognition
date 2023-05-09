import torch
import torch.nn as nn

class BNLayer(nn.Module):
    """
    Implement the batch normalization layer. 
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Please consult the documentation of batch normalization 
        (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
        for the meaning of the arguments. 
        
        Here you are asked to implementing a simpler version of the layer. Here are differences. 
        1. This implementation will be applied to a CNN's feature map with format N,C,H,W. so a channel of a 
           feature shares a set of parameters. 

        2. The initializers both arrays, so you can use them directly. (tensorflow has special initializer 
           classes)
        """

        super(BNLayer, self).__init__()

        # TODO: please complete the initialization of the class and take all these options 
        shape = (1, num_features, 1, 1)
        
        self.running_mean = torch.zeros(shape)
        self.running_var  = torch.ones(shape)
        
        self.gamma    = nn.Parameter(torch.ones(shape))
        self.beta     = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps      = eps


    def forward(self, inputs):
        """
        Implement the forward calculation during training and inference (testing) stages. 
        """
        # TODO: Please implement the caluclation of batch normalization in training and testing stages
        # NOTE 1. you can use the binary flag `self.training` to check whether the layer works in the training or testing stage:
        # NOTE 2. you need to update the moving average of mean and variance so you can use later in the inference stage 

        if self.training:
          mean = torch.mean(inputs, dim=(0, 2, 3), keepdims=True)
          var = torch.var(inputs, dim=(0, 2, 3),  keepdims=True, unbiased=False)
        else:
          mean = self.running_mean
          var = self.running_var
        outputs = self.gamma * (inputs - mean)/torch.sqrt(var + self.eps) + self.beta

        self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var
        return outputs


































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
