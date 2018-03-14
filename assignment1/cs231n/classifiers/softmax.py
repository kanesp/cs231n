import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  ss=0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)#1*10
      correct_score = scores[y[i]]
      l = -np.log(np.exp(correct_score) / np.sum(np.exp(scores)))
      loss += l

      dW[:, y[i]] -= X[i].T
      for j in xrange(num_classes):
        dW[:, j] += (np.exp(scores[j])/np.sum(np.exp(scores)))*X[i].T

  loss/=num_train
  loss += reg*np.sum(W*W)
  dW/=num_train
  dW += reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # exp_scores = np.exp(scores)
  # sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
  # norm_exp_scores = exp_scores / sum_exp_scores
  correct_score = scores[np.arange(num_train),y]
  correct_score = correct_score.reshape(num_train,1)
  a = np.sum(np.exp(scores), axis=1, keepdims=True)#(500,1)
  loss = np.sum(-np.log(np.exp(correct_score)/a))
  loss /= num_train
  loss += reg * np.sum(W * W)

  weight = np.zeros(shape=(num_train,num_classes))#500*10
  weight[np.arange(num_train)] = 1 / a[np.arange(num_train)]
  scores = np.exp(scores)
  weight = weight*scores
  np.zeros(shape=(num_train, num_classes))
  y_ture = np.zeros(shape=(num_train,num_classes))
  y_ture[np.arange(num_train),y] = 1.0
  dW += np.dot(X.T,weight-y_ture)/num_train
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW