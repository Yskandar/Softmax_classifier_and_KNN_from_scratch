import numpy as np
import pdb


class KNN(object):

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Inputs:
    - X is a numpy array of size (num_examples, D)
    - y is a numpy array of size (num_examples, )
    """
    self.X_train = X
    self.y_train = y

  def compute_distances(self, X, norm=None):
    """
    Computes the distance between each test point in X and each training point
    in self.X_train.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.
    - norm: the function with which the norm is taken.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    if norm is None:
      norm = lambda x: np.sqrt(np.sum(x**2))
      #norm = 2

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in np.arange(num_test):
        
      for j in np.arange(num_train):
        # Let us compute the distance between the ith test point and the jth       
        # training point using norm(), and store the result in dists[i, j].     
        dists[i, j] = norm(X[i] - self.X_train[j])

    return dists

  def compute_L2_distances_vectorized(self, X, norm = None):
    """
    Computes the distance between each test point in X and each training point
    in self.X_train WITHOUT using any for loops.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # Let us compute the L2 distance between the ith test point and the jth       
    # training point and store the result in dists[i, j]. We do this without any for loops


    testnorm = np.sum(X**2, axis=1)
    trainnorm = np.sum((self.X_train)**2, axis=1)
    testnorm = np.reshape(testnorm,(1, -1))
    trainnorm = np.reshape(trainnorm,(1, -1))

    dotprod = X @ np.transpose(self.X_train)

    dists = np.sqrt(np.transpose(testnorm) + trainnorm -2*dotprod)

    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predicts a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in np.arange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #   Let us select the distances to calculate and then store the labels of 
      #   the k-nearest neighbors to the ith test point.  The function
      #   numpy.argsort may be useful.
      #   
      #   After doing this, we find the most common label of the k-nearest
      #   neighbors.  Then, we store the predicted label of the ith training example
      #   as y_pred[i].  We break ties by choosing the smaller label.
  
      sorted_indices = np.argsort(dists[i])[:k]  # Sorting the indices of the k nearest neighbors
      closest_y = [self.y_train[index] for index in sorted_indices]  # getting the labels of the k nearest neighbors
      occurences = np.bincount(closest_y)  # getting the list regrouping the number of occurences of each label
      predlabel = np.min(np.where(occurences == max(occurences)))  # selecting the most common label (the smallest label of the equally most common ones)
      
      y_pred[i] = predlabel

    return y_pred
