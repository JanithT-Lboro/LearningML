"""
The code from following a tutorial to complete a neural network.
plan to make my own in near future ... 
"""

import numpy as np
import random

# centre piece is a Network class which is used to represent the neural network
"""
sizes - number of neurons in respective layers, 
  2 neurons 1st layer
  3 neurons 2nd layer
  1 neuron final layer
  --> sizes would be [2, 3, 1] --> net = Network([2,3,1])

biases and weights initialised randomly using np.random.randn
  - there are better ways to do this, but this will do for now ...

"""
class Network(object):
  def __init__(self, sizes):  
    self.sizes = sizes
    self.num_layers = len(sizes)
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
  """
  applies the following to each layer of the NN
  a' = s(w*a + b)
    a  - activations of current layer
    a' - activations of the next layer
    s  - sigmoid function
    w  - weights for current layer
    b  - bias for current layer
    
  """
  def feedforward(self, a):
    for b,w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a
  """
  using stochastic gradient descent for the training portion of the neural 
  network. 
    in general, adjust weights, train, test, 
    calculate score and gradient, repeat for a number of defined steps
  """  
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """
    Training neural network using mini-batch stochastic gradient descent.
    Training data --> tuples, (x,y). Represent input to desired output
    If test data provided, NN will evaluate against test data and print cost 
    after each epoch
      --> useful for tracking progress, but really slows down the process.
      
    epochs          - number of epochs to train for (steps)
    mini_batch_size - size of the mini_batches to use when sampling 
    eta             - the learning rate
    
    In each epoch, it starts by randomly shuffling the training data, 
    consequently partitioning it into mini-batches of the appropriate size.
      - easy way to randomly sample from the training data
    Then for each mini_batch, apply a single step of gradient descent 
      - implemented by 'self.update_mini_batch(mini_batch, eta)' 
      ... code for this follows
      - this updates NN with weights and biases according to single iteration of
      ... gradient descent
    """
    if test_data: n_test = len(test_data) # setting number of tests to length if test_data exists
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [
          training_data[k:k+mini_batch_size]
          for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),n_test))
      else:
        print("Epoch {0} complete".formate(j))
        
  def update_mini_batch(self, mini_batch, eta):
    """
    update weights and biases by applying gradient descent
    - uses backpropagation for single mini batch
    
    mini batch --> list of tuples (x,y)
    eta        --> learning rate
    
    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      - invokes backpropagation algorithm ... code seen later
      - fast way of computing the gradient of the cost function
      ... update_mini_batch works by computing the gradients for every 
          training example in mini_batch
          - and then adjusting weights and bias appropriately
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x,y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw 
                    for w, nw in zip(self.weights, nabla_w)]
    self.weights = [b-(eta/len(mini_batch))*nb 
                    for b, nb in zip(self.biases, nabla_b)]
    
    
    
    
"""
sigmoid function(s) is 1/(1+e^(-x))
derivative of sigmoid function(s), can be written as: s(1-s)
"""
def sigmoid(z):
  return 1/(1+np.exp(-z))  
def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))
