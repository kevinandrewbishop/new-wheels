from __future__ import division
import numpy as np
import pandas as pd

class ANN():
    '''
    Artificial Neural Network
    '''
    def __init__(self, hidden = 7, alpha = .5, warm_start = False):
        self.hidden = hidden
        self.alpha = alpha
        self._trained = False
        self.warm_start = warm_start

    def _initialize_matrices(self, X, y):
        if len(y.shape) == 1:
            self.out_shape = 1
        else:
            self.out_shape = y.shape[1]
        self.in_shape = X.shape[1]#to add bias term
        self.in_hid = np.random.randn(self.in_shape, self.hidden)
        self.hid_out = np.random.randn(self.hidden, self.out_shape)

    def fit(self, X, y):
        X_ = self.add_intercept(X)
        if not (self._trained and self.warm_start):
            #if it's already been trained, and warm_start is True,
            #then do not initialize weights
            self._trained = True
            best_score = float('inf')
            for i in range(20):
                #randomly initialize 20 times, choose best starting point
                #helps get closer to global optima
                self._initialize_matrices(X_, y)
                self.feed_forward(X_)
                error = self.error(y)
                print "Init Error: %s" %error
                if error < best_score:
                    best_score = error
                    best_in_hid = self.in_hid
                    best_hid_out = self.hid_out
            self.in_hid = best_in_hid
            self.hid_out = best_hid_out
            self.feed_forward(X_)
            print best_score, self.error(y)

        error = float('inf')
        decrease = True
        for i in range(200):
            self.feed_forward(X_)
            self.backprop(X_, y)
            prev_error = error
            error = self.error(y)
            prev_decrease = decrease
            decrease = error < prev_error
            print error, self.alpha
            if not decrease:
                #adaptive learning rate. If the last 2 iterations fail to
                #decrease error, divide learning rate by 5
                self.alpha/= 2


    def add_intercept(self, X):
        X_ = X.copy()
        X_['intercept'] = 1
        return X_

    def logit(self, x):
        return 1/(1+np.exp(-x))


    def feed_forward(self, X):
        self.hid_raw = X.dot(self.in_hid)
        self.hid = self.hid_raw.applymap(self.logit)
        self.out_raw = self.hid.dot(self.hid_out)
        self.out = self.out_raw.applymap(self.logit)

    def backprop(self, X, y):
        diff = -(y - self.out)
        do_di = self.out*(1-self.out)
        de_dw = self.hid.mul(do_di[0], 0).mul(diff[0], 0)
        gradient = de_dw.sum()
        gradient /= (gradient**2).sum()**.5
        gradient *= self.alpha
        self.hid_out -= gradient.values.reshape((self.hidden, 1))

        di_do = self.hid_out
        do_di2 = self.hid*(1-self.hid)
        di2_dw2 = X
        gradient = []
        for col in X.columns:
            temp = do_di2.mul(di2_dw2[col],0)
            temp *= di_do.T[0] #not good
            temp = temp.mul(do_di[0], 0).mul(diff[0], 0)
            temp = temp.sum().sum()
            gradient.append(temp)
        gradient = pd.Series(gradient)
        gradient /= (gradient**2).sum()**.5
        gradient *= self.alpha
        self.in_hid -= gradient.values.reshape((self.in_shape,1))

    def error(self, y):
        return (((y - self.out)**2).mean()**.5)[0] #root mean squared error

    def predict(self, X):
        X_ = self.add_intercept(X)
        self.feed_forward(X_)
        return (self.out[0] > 0.5).astype(int)


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics

    middle_neurons = 5

    X = pd.DataFrame(np.random.randn(1000,2), columns = ['x1', 'x2'])
    y = ((-2*X['x1'] + X['x2'] + X['x1']**3)>0)*1
    y = y.reshape((1000,1))
    ann = ANN(middle_neurons, warm_start = False)
    ann.fit(X, y)
    nn = MLPClassifier((middle_neurons,), 'logistic')
    nn.fit(X, y)
