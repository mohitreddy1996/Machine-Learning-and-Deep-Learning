import numpy as np
from random import shuffle

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
  # modifying W and X for calculations.
  W = W.T
  X = X.T
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # get number of classes and number of training elements.
  num_classes = W.shape[0]
  num_train = X.shape[1]

  # compute for all of them.
  for i in range(num_train):
    # vector scoring
    f_score = W.dot(X[:, i])

    # normalization technique. Get max and subtract from all others.
    norm_tech = np.max(f_score)
    f_score -= norm_tech

    # compute loss 
    # loss = -f_score[y] + log(sum(e^score[f_scoreij]))
    temp_sum = 0.0
    for f_score_ij in f_score:
      temp_sum += np.exp(f_score_ij)
    loss += -f_score[y[i]] + np.log(temp_sum)

    # gradient calculation. Check with resources.
    for j in range(num_classes):
      p = np.exp(f_score[j])/temp_sum
      dW[j, :] += (p - (j==y[i]))*X[:, i]

  # compute average and regularization.
  loss /=num_train
  dW /=num_train

  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW.T


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # computation sake
  W = W.T
  X = X.T
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # get num of classes and training number.
  num_classes = W.shape[0]
  num_train = X.shape[1]

  # compute function scores.
  f_score = np.dot(W, X)

  # normalisation technique
  norm_tech = np.max(f_score)
  f_score -= norm_tech

  # calculate the loss function.
  f_score_mod = f_score[y, range(num_train)]
  loss = -np.mean(np.log(np.exp(f_score_mod)/np.sum(np.exp(f_score))))

  # gradient. check.. (Referred online!!!!)
  p = np.exp(f_score)/np.sum(np.exp(f_score), axis = 0)
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((p-ind), X.T)
  dW /= num_train

  # regularization.
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW.T

