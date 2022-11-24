import numpy as np
import utils
import os
import pickle
import matplotlib.pyplot as plt

class Simple_Decision_Tree:
    def predict_utils(self,x): 
        if x[0] >= -80.248086:
            return 0
        else:
            if x[1] >= 37.695206:
                return 0
            else:
                return 1
    def predict(self,X): 
        n = len(X)
        y = np.zeros(n, dtype=np.int8) 
        for i in range(n):
            y[i] = self.predict_utils(X[i,:])
        return y



if __name__ == "__main__":
    with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

    X = dataset["X"]
    y = dataset["y"]
    model = Simple_Decision_Tree()
    utils.plotClassifier(model, X, y)
    fname = os.path.join("..", "figs", "simple_decision.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)