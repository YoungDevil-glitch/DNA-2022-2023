import numpy as np
import utils

class DecisionStumpErrorRate:

    def __init__(self):
        self._minError = None
        self._splitVariable = None
        self._splitValue = None
        self._splitSat = None
        self._splitNot =None

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

        self._minError = utils.loss_l0(y_mode*np.ones(N),y)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                condition = X[:,d] >= value
                y_sat = utils.mode(y[condition])
                y_not = utils.mode(y[np.logical_not(condition)])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[np.logical_not(condition)] = y_not

                # Compute error
                errors = utils.loss_l0(y_pred,y)
                # Compare to minimum error so far
                if errors < self._minError:
                    # This is the lowest error, store this value
                    self._minError = errors
                    self._splitVariable = d
                    self._splitValue = value
                    self._splitSat = y_sat
                    self._splitNot = y_not

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
