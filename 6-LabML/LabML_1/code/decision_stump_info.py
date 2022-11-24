import numpy as np
import utils

def entropy(p):
    plogp = [0]*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)

# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?

class DecisionStumpInfoGain:
    def __init__(self, loss = entropy):
        self._minError = None
        self._splitVariable = None
        self._splitValue = None
        self._splitSat = None
        self._splitNot =None
        self._loss = loss
        self._info  = None

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self._splitSat = y_mode

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        self._minError = utils.loss_l0(y_mode * np.ones(N),y)
        h_0 = self._loss(count/sum(count))
        self._info = 0
        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                condition = X[:,d] >= value
                y_sat = utils.mode(y[condition])
                y_not = utils.mode(y[np.logical_not(condition)])
                count_1 = np.bincount(y[condition])
                count_2 = np.bincount(y[np.logical_not(condition)])
                h_1 = self._loss(count_1/sum(count_1))
                h_2 = self._loss(count_2/sum(count_2))

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[y[np.logical_not(condition)]] = y_not
                # Compute error
                info = h_0 - h_1 *(sum(count_1)/N) - h_2 *(sum(count_2)/N)
                errors = utils.loss_l0(y_pred,y)
                # Compare to minimum error so far
                if info > self._info:
                    # This is the lowest error, store this value
                    self._minError = errors
                    self._splitVariable = d
                    self._splitValue = value
                    self._splitSat = y_sat
                    self._splitNot = y_not
                    self._info = info

    def predict(self, X):
        N, D = X.shape

        if self._splitVariable is None:
            return self._splitSat * np.ones(N)

        yhat = np.zeros(N)

        for m in range(N):
            if X[m, self._splitVariable] >= self._splitValue:
                yhat[m] = self._splitSat
            else:
                yhat[m] = self._splitNot

        return yhat
    


    
"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
