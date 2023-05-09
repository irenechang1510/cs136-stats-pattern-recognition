"""
Helper functions for getting parameters from RNN models (a conventional RNN and a GRU). The function reads 
weight matrices within RNN layers and the dispatch their values to corresponding weights or biases for 
the calculation of gates and hidden states. 
"""

import numpy as np
import torch
from copy import deepcopy

def get_rnn_params(rnn_layer):
    """Get parameters from an RNN layer
    
    inputs: 
        rnn_layer: a torch RNN layer 
    outputs: 
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
    """
    
    # ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
    with torch.no_grad():
        wt_x, wt_h, bias_ih, bias_hh = rnn_layer._flat_weights
        wt_x = deepcopy(wt_x.numpy()).T
        wt_h = deepcopy(wt_h.numpy()).T
        bias = (bias_ih + bias_hh).numpy()
    return wt_h, wt_x, bias

def get_gru_params(gru_layer):
    """Get parameters from a GRU layer 
    
    args: 
        gru_layer: a torch GRU layer
    returns: 
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
    """
    with torch.no_grad():
        wt_x, wt_h, bias_ih, bias_hh = gru_layer._flat_weights

        hidden_size = wt_h.shape[1]
        wt_x = deepcopy(wt_x.numpy()).T
        wt_h = deepcopy(wt_h.numpy()).T

        biasir, biasiz, biasin = np.split(deepcopy(bias_ih.numpy()), 3) 
        biashr, biashz, biashn = np.split(deepcopy(bias_hh.numpy()), 3)
        
        wt_ir, wt_iz, wt_in = np.split(wt_x, 3, axis=1) 
        wt_hr, wt_hz, wt_hn = np.split(wt_h, 3, axis=1)

        linear_trans_r = (wt_ir, biasir, wt_hr, biashr)
        linear_trans_z = (wt_iz, biasiz, wt_hz, biashz)
        linear_trans_n = (wt_in, biasin, wt_hn, biashn)
        
    return linear_trans_r, linear_trans_z, linear_trans_n



def get_mha_params(mha_layer):
    """Get parameters from a Multi-Head Attention (MHA) layer 
    
    inputs: 
        mha_layer: a torch MHA layer. Please first check the documentation of `torch.nn.MultiheadAttention`
    outputs: 
        Wq: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wk: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wv: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wo: a numpy tensor with shape [embed_dim, embed_dim]
    """

    
    Wq_packed, Wk_packed, Wv_packed = np.split(deepcopy(mha_layer.in_proj_weight.detach().T.numpy()), 3, axis=1)


    Wq = np.split(Wq_packed, mha_layer.num_heads, axis=1)
    Wk = np.split(Wk_packed, mha_layer.num_heads, axis=1)
    Wv = np.split(Wv_packed, mha_layer.num_heads, axis=1)

    Wo = deepcopy(mha_layer.out_proj.weight.detach().numpy().T)

    
    return Wq, Wk, Wv, Wo






