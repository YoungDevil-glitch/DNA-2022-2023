# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              
import pandas as pd                             
import matplotlib.pyplot as plt                 
from scipy.optimize import approx_fprime        
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`


""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

import utils
from decision_stump import DecisionStumpEquality
from decision_stump_generic import DecisionStumpEqualityGeneric
from decision_stump_error import DecisionStumpErrorRate
from decision_stump_info import DecisionStumpInfoGain, entropy
from decision_tree import DecisionTree



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # YOUR ANSWER HERE
        print("The minimum depth of a binary tree with 64 leaf nodes is: 6")
        print("The minimum depth of binary tree with 64 nodes (includes leaves and all other nodes) is: 6") 
    
    elif question == "1.2":
        # YOUR ANSWER HERE
        print(f"The running time of the function", "func1 ", "is: O(N)")
        print("The running time of the function", "func2 ", "is: O(N)")
        print("The running time of the function", "func3 ", "is: O(1)")
        print("The running time of the function", "func4 ", "is: O(N²)")
    
    elif question == "2.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values
        print(names)
        m =len(max(names, key = len))
        # YOUR CODE HERE
        space_1 =" "*(m//2)
        space_3 =" "*(m-m//2)
        space_2 =" "*6
        h = f"|{space_1} Colum {space_3}| min {space_2}| max{space_2}| mean{space_2}| median{space_2}| mode{space_2}|" 
        x = "-"*(len(h)//2)
        print(h)
        print(x) 
        for name in names:
            k = m+7 -len(name)
            start = " "*(k//2)
            end = " "*(k -k//2)
            name_max = df[name].max()
            name_min = df[name].min()
            name_mean = df[name].mean()
            name_median = df[name].median()
            name_mode = utils.mode(df[name].values)
            to_string = f'|{start}{name}{end}| {name_min:9.5f} | {name_max:9.5f} | {name_mean:9.5f} | {name_median:9.5f} | {name_mode:9.5f} |' 
            print(to_string)
        print(x)
        print("Mode is not a good statistic for continuous variable. Instead, we regroup the data in bins (value is remplaced by the bins that include it) and the determine the mode of the bins")
        h = f"|{space_1} Colum {space_3}| 5% {space_2}| 25%{space_2}| 50%{space_2}| 75%{space_2}| 95%{space_2}|" 
        x = "-"*len(h)
        print(h)
        print(x) 
        for name in names:
            k = m+7 -len(name)
            start = " "*(k//2)
            end = " "*(k -k//2)
            name_5 = df[name].quantile(0.05)
            name_25 = df[name].quantile(0.25)
            name_50 = df[name].quantile(0.5)
            name_75 = df[name].quantile(0.75)
            name_95 = df[name].quantile(0.95)
            to_string = f'|{start}{name}{end}| {name_5:9.5f} | {name_25:9.5f} | {name_50:9.5f} | {name_75:9.5f} | {name_95:9.5f} |'
            print(to_string)
        print(x)
        dict_names = {}
        for name in names:
            dict_names[name] = [ df[name].mean(),df[name].std()]
        name_1 = max(dict_names , key = lambda x: x[0])
        name_2 = max(dict_names , key = lambda x: x[1])
        name_3 = min(dict_names , key = lambda x: x[0])
        name_4 = min(dict_names , key = lambda x: x[1])
        print(f"La variable moyenne maximale est {name_1}")
        print(f"La variable moyenne minimale est {name_3}")
        print(f"La variable variance maximale est {name_2}")
        print(f"La variable variance minimale est {name_4}")
    elif question == "2.2":
        
        # YOUR CODE HERE : modify HERE
        figure_dic = {'A':[4, "This is a line plot"],
                      'B':[3, "The form of a boxplot"],
                      'C':[2, "No legend"],
                      'D':[1, "The legend contain the name of every region"],
                      'E':[6, "Some observation are far from a regression line"],
                      'F':[5, "The points seem to form a line"]}
        for label in "ABCDEF":
            print("Match the plot", label, "with the description number: ",figure_dic[label][0])
            print("Explication:",figure_dic[label][1])
        
    elif question == "3.1":
        # 1: Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3: Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with Equality rule error: %.3f"
              % error)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_1_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # YOUR ANSWER HERE
        print("It makes sense to use an  equality-based splitting rule rather than the threshold-based splits when we deal with categorical variable" )
    
    elif question == "3.2":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]
        loss = utils.loss_l0
        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = loss(y_pred, y)
        print("Mode predictor error: %.3f" % error)

        # 2: Evaluate the generic decision stump
        model = DecisionStumpEqualityGeneric()
        y_pred = model.fit_predict(X, y)

        error = model.score(X,y) 
        print("Decision Stump with Equality Generic rule error: %.3f"
              % error)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        

        
            # YOUR CODE HERE

        print("Decision Stump Generic rule error: %.3f" % error)

        # 3: Plot result
            # YOUR CODE HERE


    elif question == "3.3":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]
        loss = utils.loss_l0
        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = loss(y_pred, y)
        print("Mode predictor error: %.3f" % error)

        # 2: Evaluate the generic decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)
        error = loss(y_pred, y)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
          # 1: Load citiesSmall dataset         
            # YOUR CODE HERE

        # 2: Evaluate the inequality decision stump
            # YOUR CODE HERE

        print("Decision Stump with inequality rule error: %.3f" % error)

        # 3: Plot result
            # YOUR CODE HERE
               
    elif question == "3.4":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]
        loss = utils.loss_l0
        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = loss(y_pred, y)
        print("Mode predictor error: %.3f" % error)

        # 2: Evaluate the generic decision stump
        model = DecisionStumpInfoGain(loss = entropy)
        model.fit(X, y)
        y_pred = model.predict(X)
        error = loss(y_pred, y)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        print("Decision Stump with info gain rule error: %.3f" % error)

        # 3: Plot result
            # YOUR CODE HERE

    
    elif question == "3.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q3_5_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # Here YOUR CODE
        print("The code corresping to this model is in the python file DecisionTree.py")

    elif question == "3.6":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try
       
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="errorrate")
        
        
        t = time.time()
        my_tree_errors_infogain = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_6_tree_errors.pdf")
        plt.savefig(fname)
        comment = """ The tree built using the error rate was underperforming compared to the other methods.
        It started as the better option but plateaued rapidly. Using information gain seem to be a better 
        strategy. The error_1 approach overfit the model at small depth that leads to imprecision at biggerdepth.
        """
        
        n = my_tree_errors_infogain.argmin()
        print(n)
        model = DecisionTree(max_depth=n,stump_class=DecisionStumpInfoGain)
        model.fit(X, y) 
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q3_6_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        

    else:
        print("No code to run for question", question)