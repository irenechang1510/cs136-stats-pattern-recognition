# Comp137 DNN: Assiginment 4 

## Objectives
Through this project, you will 

1. learn the internal structure of vanilla RNNs and GRUs;
1. learn the internal structure of MHA;
3. gain the experience of training RNN/Transformer models.

## Instructions 

This assignment has two tasks. The first one is to implement the forward calculation of RNNs (the conventional one and the GRU) and a Multi-Head Attention layer. You will need to run `rnn.ipynb`, and you will see detailed instructions there.  To support the notebook, you will need to implement all functions in `implementation.py`. 

The second one is to model language with an RNN or a Transformer. In this task, you will implement an RNN model or a Transformer that predicts the next character in a sentence. The goal is to improve the predictive accuracy of the model, which is measured by the per-character cross-entropy loss. The model can also generate text. The generated text is also an evidence whether the model captures the pattern of the language. Please run `sentence_modeling.ipynb` and follow instructions there. 

## Grading

Detailed grading rules are inside the notebook. Finally, you need to submit 
* all files in this folder;
* the trained model
* the code you have written for your own language model

To submit these files in a zip file, please zip the folder `assignment4` into a zip file.

Your implementations and your trained model will be graded by the autograder through GradeScope.

When we grade your code, we check your result notebook as well as your code. If your code cannot generate the result of a problem in your notebook file, you will get zero point for that problem. Actually, if you follow instructions in the notebook and finish all tasks, you are likely to get most points. 


