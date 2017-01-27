'''
Wanted to try creating a simple pandas-based K Nearest Neighbor classifier.
Very brute force and somehow needs to be sped up.

One way to speed it up:
Sort the training X data by whichever feature has the largest variance. Let's call this feature F.
(Of course, sort the training Y data as well so that it is still properly aligned with the train X data).
Then for each test X point p, find the row R' of the training data where test F is as close as possible
to train F. For example, perhaps in row 121 train F is most similar to test F.
Measure the distance between test point p and the train point at row R'. Increment the row number by one
and measure the distance between test point p and the train point (now at row R' + 1). Keep a list L of length k
containing the distances you're measuring and the rows they correspond to. As you keep incrementing the row
number, if at any point the distance between test F and train F is larger than the largest distance in your
list L, you know you have no need to check the distances of the remaining training observations. This is
because if these observations are further away than the k furthest neighbor in dimension F alone, then they 
will certainly be further when you include all the remaining dimensions besides F.
In addition to incrementing forward you will have to increment backward from R' to R'-1. Probably the most
efficient way to do this is to toggle back and forth between positive and negative increments. Specifically,
you will evaluate in this order: R'; R'+1; R'-1; R'+2; R'-2... until you hit the stopping criteria mentioned above.

Identifying R' efficiently will probably require indexing F in the training data. For example, you could store
the row number and value of F's deciles. Then find the decile that most closely matches the test point being
evaluated and then search from there. Maybe binary search fits in here somehow?
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
