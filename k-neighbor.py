'''
Wanted to try creating a simple pandas-based K Nearest Neighbor classifier.
Very brute force and somehow needs to be sped up.
'''

import matplotlib.pyplot as plt
import pandas as pd


class NearestNeighbor():
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X, k):
        prediction = X.apply(lambda x: self._get_nearest_Y_mean(x, k), 1)
        return prediction

    def _nearest_index(self, X, k):
        output = self.X - list(X.T)
        output = output**2
        output = output.sum(1)
        output.sort()
        output = list(output.index)
        return output[:k]

    def _get_nearest_Y_mean(self, X, k):
        index = self._nearest_index(X, k)
        output = self.Y[index]
        output = output.mean()
        return output


if __name__ == '__main__':
    X = pd.DataFrame.from_csv('test KNN data.txt', sep = '\t')
    Y = X['Y']
    X = X[['X%s' %i for i in range(1,5)]]
    
    half = X.shape[0]//2
    X_train = X[:half]
    Y_train = Y[:half]
    X_test = X[-half:]
    Y_test = Y[-half:]
    
    clf = NearestNeighbor()
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test, 5)
    
    output = pd.DataFrame(prediction, columns = ['prediction'])
    output['Y'] = Y_test
    
    output.plot(kind = 'scatter', x = 'prediction', y = 'Y')
    
    plt.show()
