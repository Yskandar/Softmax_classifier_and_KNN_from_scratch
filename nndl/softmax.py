import numpy as np


class Softmax(object):

  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
    Initializes the weight matrix of the Softmax classifier.  
    Note that it has shape (C, D) where C is the number of 
    classes and D is the feature size.
    """
    self.W = np.random.normal(size=dims) * 0.0001

  def loss(self, X, y):
    """
    Calculates the softmax loss.
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
  
    Returns a tuple of:
    - loss as single float
    """

    # Initialize the loss to zero.
    loss = 0.0

    # Let us calculate the normalized softmax loss.
    N, D = np.shape(X)
    C, D = np.shape(self.W)

    for i in range(N):
      loss_temp = 0
      pred_correct_class = np.exp(self.W[y[i]] @ np.transpose(X[i]))
      pred_rest = 0
      for j in range(C):
        pred_rest += np.exp(self.W[j] @ np.transpose(X[i]))
      loss_temp = np.log(pred_correct_class/pred_rest)
      loss += -loss_temp/N


    return loss

  def loss_and_grad(self, X, y):
    """
    Same as self.loss(X, y), except that it also returns the gradient.

    Output: grad -- a matrix of the same dimensions as W containing 
      the gradient of the loss with respect to W.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    grad = np.zeros_like(self.W)
  
    # Let us calculate the softmax loss and the gradient.
    N, D = np.shape(X)
    C, D = np.shape(self.W)
    for i in range(N):
      loss_temp = 0
      pred_correct_class = np.exp(self.W[y[i]] @ np.transpose(X[i]))
      pred_rest = 0
      for j in range(C):  # computing the rest of the probabilities
        pred_rest += np.exp(self.W[j] @ np.transpose(X[i]))
      loss_temp = np.log(pred_correct_class/pred_rest)
      loss += -loss_temp/N
    
    for k in range(C):
      grad_temp = 0
      for i in range(N):
        loss_temp = 0
        pred_correct_class = np.exp(self.W[k] @ np.transpose(X[i]))
        pred_rest = 0
        for j in range(C):  # computing the rest of the probabilities
          pred_rest += np.exp(self.W[j] @ np.transpose(X[i]))
        F = pred_correct_class/pred_rest 
        grad_temp += (F - int(y[i] == k))*X[i]
      grad[k] = grad_temp/N

    return loss, grad

  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])
  
      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
    inputs and ouptuts as loss_and_grad.
    """
    loss = 0.0
    grad = np.zeros(self.W.shape) # initialize the gradient as zero
  
    # Let us calculate the softmax loss and gradient WITHOUT any for loops.
    C, D = np.shape(self.W)
    N, D = np.shape(X)
    alpha = self.W @ np.transpose(X)  # getting all the scores
    maximums = np.max(alpha, axis=0)
    alpha = np.exp(alpha - maximums)  # Normalizing so as to avoid overflow
    S = (np.ones((1,C)) @ alpha)  # summing the scores
    E = alpha[np.transpose(y), np.arange(N)]  # getting the scores of the correct class for each sample
    loss = np.sum((-1/N)*np.log(E/S))  # computing the loss
    

    E2 = (alpha/S)  # getting the scores of each sample for each class divided by the sum of the scores
    Count = np.transpose([np.arange(C)] * N)  # The three next operations find where k == y(i)
    R = Count - np.transpose(y)
    Adjust = np.where(R == 0, 1, 0)
    E3 = E2 - Adjust  # subtracting 1 where k == y(i)
    grad = (1/N)*E3 @ X  # weighted sum of the samples by the associated softmax score

    return loss, grad

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

    self.init_weights(dims=[np.max(y) + 1, X.shape[1]])	# initializes the weights of self.W

    # Run stochastic gradient descent to optimize W
    loss_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      # Let us sample batch_size elements from the training data for use in 
      # gradient descent.

      indices = np.random.choice(num_train, batch_size)
      X_batch, y_batch = X[indices], y[indices]

      # evaluate loss and gradient
      loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
      loss_history.append(loss)

      # Gradient step
      self.W = self.W - (learning_rate * grad)

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])

    # Predicts the labels given the training data.
    C, D = np.shape(self.W)
    alpha = np.exp(self.W @ np.transpose(X))
    S = np.ones((1,C)) @ alpha  # getting the sum of the rest of the probabilities
    scores = np.transpose(alpha/S)
    y_pred = np.array([np.argmax(row) for row in scores])

    return y_pred

