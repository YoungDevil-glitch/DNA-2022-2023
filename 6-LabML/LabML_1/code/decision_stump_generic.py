import numpy as np
import utils



class DecisionStumpEqualityGeneric:
    '''
    The goal of this class is to implement a generic version de the stump equality in terms of:
    - Objective hyperparameter : any user defined method for computing the error
    - Scoring function : reporting directly the score of validation
     
    Inheritance from DecisionStumpEquality class can be also a good idea
    '''

    def __init__(self, loss=utils.loss_l0):
        self._minError = None
        self._splitVariable = None
        self._splitValue = None
        self._splitSat = None
        self._splitNot =None 
        self._splitLoss = loss      
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
            self._minError = 0
            return

        self._minError = self._splitLoss(y_mode*np.ones(N),y)

        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = self._splitLoss(y_pred,y)

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
        X = np.round(X)

        if self._splitVariable is None:
            return self._splitSat * np.ones(N)

        yhat = np.zeros(N)

        for m in range(N):
            if X[m, self._splitVariable] == self._splitValue:
                yhat[m] = self._splitSat
            else:
                yhat[m] = self._splitNot

        return yhat

    def fit_predict(self,X,y):
        self.fit(X,y)
        return self.predict(X)
            
    def score(self, X, y):
        y_pred = self.predict(X)
        return self._splitLoss(y_pred,y)


