import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class IrisDF:
    def __init__(self, normalize=True, test_size=0.3):
        iris = datasets.load_iris()
        X = None
        if normalize:
            X = preprocessing.normalize(iris.data) 
        else:
            X = iris.data
        y = self.encode(iris.target)

        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size = test_size)

    def encode(self, data):
        y = []
        for i in data:
            if i == 0:
                y.append([0, 0, 1])
            elif i == 1:
                y.append([0, 1, 0])
            else:
                y.append([1, 0, 0])
        return np.array(y)