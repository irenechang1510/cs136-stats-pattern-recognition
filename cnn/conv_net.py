################################################################################################
from torch.nn import Sequential
from torch.nn import Conv2d, MaxPool2d, Dropout2d, Flatten, Linear, ReLU, BatchNorm2d, Sequential, AvgPool1d, Softmax, LazyLinear

# NOTE: you should NOT import anything else from torch or other deep learning packages. It 
# means you need to construct a CNN using these layers. If you need other layers, you can ask us 
# first, but you CANNOT use existing models such as ResNet from tensorflow. 
################################################################################################

def ConvNet(**kwargs):
    """
    Construct a CNN using `torch`: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html. 
    """
    
    # TODO: implement your own model

    # This is a very simple model as an example, which sums up all pixels and does classification 
    model = Sequential(*[Conv2d(3, 64, [3, 3]), # Input channel is 3, output channel is 64
                        BatchNorm2d(64),
                        ReLU(),

                        Conv2d(64, 128, [2, 2], stride = (2,2)),
                        BatchNorm2d(128),
                        ReLU(),
                        MaxPool2d(2), 

                        Conv2d(128, 256, [2, 2], stride = (2,2)),
                        ReLU(),
                        BatchNorm2d(256),
                        MaxPool2d(2),

                        Conv2d(256 ,512, [2, 2], stride = (1,1), padding='same'),
                        ReLU(),
                        BatchNorm2d(512),
                        MaxPool2d(2), 

                        Flatten(2, -1), # N,C,H, W => N,C, H * W
                        AvgPool1d(5), # N, C, L => N, C, L'
                        Flatten(1,-1), # N, C, L' => N, L''
                        LazyLinear(3),
                       ])

    return model 



































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
