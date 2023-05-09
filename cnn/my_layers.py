import numpy as np

# NOTE: it is easier for you to implement the convolution operation and the pooling operation with 
#       for-loops. Backpropagation and speed are not our considerations in this task.
#       

def conv_forward(input, filters, bias, stride, padding):
    """
    An implementation of the forward pass of the convolutional operation. 
    Please consult the documentation of `torch.nn.functional.conv2d` 
    [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)
    for the calculation and arguments for this operation. 
    
    We are considering a simpler case: the `input` is always in the format "NCHW". 
    We only consider two padding cases: "SAME" and "VALID". 

    """
    # TODO: Please implement the forward pass of the convolutional operation   
    # NOTE: you will need to compute a few sizes -- a handdrawing is useful for you to do the calculation.  
    n_f, n_c_f, f_h, f_w = filters.shape 
    N, n_c, in_dim_h, in_dim_w = input.shape
    if padding == 'valid':
      out_dim_h = int((in_dim_h - f_h)/stride[0])+1
      out_dim_w = int((in_dim_w - f_w)/stride[1])+1
      out = np.zeros((N, n_f,out_dim_h,out_dim_w))
    
    elif padding == 'same':
      pad_h = f_h - 1
      pad_w = f_w - 1

      out = np.zeros((N, n_f, in_dim_h, in_dim_w))

      input = np.append(
        arr = np.zeros((N, n_c, int(np.ceil(pad_h/2)), in_dim_w)), values = input, axis=2)
      input = np.append(
        arr = input, values = np.zeros((N, n_c, int(np.floor(pad_h/2)), in_dim_w)), axis=2)

      N, n_c, in_dim_h, in_dim_w = input.shape
      input = np.append(
        arr= np.zeros((N, n_c, in_dim_h, int(np.ceil(pad_w/2)))), values = input, axis=3)
      input = np.append(
        arr = input, values = np.zeros((N, n_c, in_dim_h, int(np.floor(pad_w/2)))), axis=3)
      
      N, n_c, in_dim_h, in_dim_w = input.shape

    for curr_f in range(n_f):
      curr_y = out_y = 0
      
      while curr_y + f_h <= in_dim_h:
        curr_x = out_x = 0
          
        while curr_x + f_w <= in_dim_w:
          out[0, curr_f, out_y, out_x] = np.sum(
            filters[curr_f] * input[:, :,curr_y:curr_y+f_h, curr_x:curr_x+f_w]) + bias[curr_f]
          curr_x += stride[1]
          out_x += 1
        curr_y += stride[0]
        out_y += 1
        
    return out



def max_pool_forward(input, ksize, stride, padding = "VALID"): # No need for padding argument here
    """
    An implementation of the forward pass of the max-pooling operation. 
    Please consult the documentation of `torch.nn.MaxPool2d` 
    [link](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    for the calculation and arguments for this operation. 
    
    We are considering a simpler case: the `input` is always in the format "NCHW". 
    We only consider two padding cases: "SAME" and "VALID". 


    """

    # TODO: Please implement the forward pass of the max-pooling operation   
    N, n_c, h_prev, w_prev = input.shape
    
    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - ksize[0])/stride[0])+1 
    w = int((w_prev - ksize[1])/stride[1])+1
    
    # create a matrix to hold the values of the maxpooling operation.
    out = np.zeros((N, n_c, h, w)) 
    
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for n in range(N):
      for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + ksize[0] <= h_prev:
          curr_x = out_x = 0
          # slide the max pooling window horizontally across the image
          while curr_x + ksize[1] <= w_prev:
              # choose the maximum value within the window at each step and store it to the output matrix
            out[n, i, out_y, out_x] = np.max(input[n, i, curr_y:curr_y+ksize[0], curr_x:curr_x+ksize[1]])
            curr_x += stride[1]
            out_x += 1
          curr_y += stride[0]
          out_y += 1
      
    return out



































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
