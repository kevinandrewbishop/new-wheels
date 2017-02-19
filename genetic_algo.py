'''
Genetic Algorithm to solve linear regression.

Naturally gradient descent would be faster, but of course this is just
for learning purposes.

The algorithm works by:
1. Randomly create population of candidate parameters
2. Evaluate fitness of each and use parent selection method
3. Create children using crossover method
4. Apply any random mutations to children
5. Apply survivor selection to see which parameters "die"
6. Repeat steps 2-5 until convergence
'''
import pandas as pd
from sklearn import metrics
from random import sample
np = pd.np

class GARegressor():
    def __init__(self, fit_intercept = True, pop_size = 1000, p_mutate = 0.1, iterations = 200, error_func = None):
        '''
        Genetic Algorithm used for linear regression. Although linear regression
        is more efficiently solved by gradient descent, the purpose was just to
        learn by building a GA.

        Parameters
        ----------
        fit_intercept: boolean, default True
            Whether to calculate the intercept in the model.
        pop_size: int, default 1000
            The size of the population that will evolve. Larger populations are
            more likely to find a better answer, but are more computationally
            expensive.
        p_mutate: float between 0 and 1, default 0.1
            The probability of a parameter mutating.
        iterations: int, default 200
            The number of reproduction/death cycles to run before stopping.
            Higher numbers are likely to converge on a better solution but take
            longer to compute.
        error_func: None or a user-defined function, default None
            Defines the fitness of a set of parameters. Lower values are fitter.
            The function must take the parameters (X, params, y).
            If None, by default it returns mean squared error.

        Examples
        --------
        >>> X = pd.read_csv('X_data_file.csv')
        >>> y = pd.read_csv('y_data_file.csv')
        >>> ga = GARegressor(p_mutate = .05)
        >>> ga.fit(X, y)
        >>> yhat = ga.predict(X)
        >>> train_error = ((y - yhat)**2).mean()
        '''
        self.fit_intercept = fit_intercept
        self.pop_size = pop_size
        self.p_mutate = p_mutate
        self.iterations = iterations
        if error_func:
            self.get_error = error_func
        self.best_score = list()
        self.avg_score = list()
        self.pops = list()


    def fit(self, X, y):
        '''
        Fit the linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples,]
            Target values

        Returns
        -------
        None. However, sets values for self.params_ and self.intercept_ to be
        used in prediction.

        '''
        if self.fit_intercept:
            X = X.copy()
            X['intercept_'] = 1
        self.initialize_pop(X.shape[1])
        self.fitness_ = np.array(self.get_fitness(X, y, self.pop))
        #Main Loop
        for i in range(self.iterations):
            self.run(X, y)
            #print "Round: %s\tError: %s" %(i, self.fitness_.min())
            if i%10 == 0:
                self.best_score.append(self.fitness_.min())
                self.avg_score.append(self.fitness_.mean())
                self.pops.append(self.pop)
        fitness = self.get_fitness(X, y, self.pop)
        best = fitness.index(min(fitness))
        self.params_ = self.pop[best]
        if self.fit_intercept:
            self.intercept_ = self.params_[-1]
            self.params_ = self.params_[:-1]
        else:
            self.intercept_ = False

    def run(self, X, y):
        '''
        Runs a single cycle of reproduction, mutation, and survival. Meant
        to be repeatedly called by the fit method.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples,]
            Target values

        Returns
        -------
        None. However, updates values for self.pop and self.fitness_.
        '''
        #fitness = self.get_fitness(X, y)
        fitness = list(self.fitness_)
        par_ind = self.select_parents(fitness, 5, 50)
        parents = self.pop[par_ind]
        children = self.create_children(parents)
        surv_index = self.select_survivors(fitness, 5, children.shape[0])
        survivors = self.pop[surv_index]
        self.pop = np.concatenate((survivors, children))
        self.fitness_ = self.fitness_[surv_index]
        child_fitness = self.get_fitness(X, y, children)
        child_fitness = np.array(child_fitness)
        self.fitness_ = np.concatenate((self.fitness_, child_fitness))


    def predict(self, X):
        '''
        Predict using linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        Returns
        -------
        prediction: a Pandas Series of predicted values.
        '''
        return X.dot(self.params_) + self.intercept_

    def initialize_pop(self, num_vars):
        '''
        Creates the initial randomized parameters that make up the population.

        Parameters
        ----------
        num_vars : int
            The number of predictive variables in X.

        Returns
        -------
        None. However, it sets the value of self.pop to a numpy matrix of
            size [pop_size, num_vars].
        '''
        self.pop = np.random.uniform(-20,20,(self.pop_size, num_vars))

    def get_error(self, X, params, y):
        '''
        Calculates the fitness of a set of parameters. In the case of linear
        regression this is the mean squared error.

        Parameters
        ----------
        X: numpy array of shape [n_samples,n_features]
            Training data
        params: numpy array of shape [n_features,]
            Linear regression coefficients.
        y : numpy array of shape [n_samples,]
            Target values

        Returns
        -------
        err: float, the mean squared error of the model
        '''
        yhat = X.dot(params)
        err = (y-yhat)**2 #sum of squared error
        err = err.mean() #measure
        return err


    def get_fitness(self, X, y, params):
        '''
        Calculates the fitness of a population of parameters. In the case of
        linear regression this is the mean squared error. This simply repeatedly
        calls get_error on each member

        Parameters
        ----------
        X: numpy array of shape [n_samples,n_features]
            Training data
        params: numpy array of shape [n_members, n_features]
            Linear regression coefficients of each member of the population.
        y : numpy array of shape [n_samples,]
            Target values

        Returns
        -------
        fitness: list of floats, the mean squared error of each member of the
        population.
        '''
        fitness = list()
        for p in params:
            fitness.append(self.get_error(X, p, y))
        return fitness

    def select_parents(self, fitness_scores, k, n):
        '''
        Selects n members of the population to be parents (i.e. to reproduce).

        Parameters
        ----------
        fitness_scores: list
            A list of fitness scores of the population. Lower scores are better.
        k: int
            The number of members of the population to enter each tournament.
            Smaller numbers give weaker members a higher chance of becoming
            parents. Higher values may reduce genetic diversity prematurely.
        n: int
            The number of members to become parents. Same as the number of
            tournaments to run.

        Returns
        -------
        parent_index: a list containing the index of parents.

        Notes
        -----
        See: https://en.wikipedia.org/wiki/Tournament_selection
        '''
        parents = list()
        for i in range(n):
            potentials = sample(fitness_scores, k)
            best = min(potentials)
            index = fitness_scores.index(best)
            parents.append(index)
        return parents

    def create_children(self, parents):
        '''
        [Long explanation, but it took some thinking to work through so I'm not
        sure how to shorten this without being confusing.]

        Creates children via gene crossover from parents plus random mutation.
        Creates one child per pair of parents. Parents at index 0 and 1 breed,
        at 2 and 3 breed, at 4 and 5 breed, and so on. Each gene has a 50%
        chance of coming from either parent and is calculated independently. So
        it is possible a child will inherit genes from only one parent.

        To compute crossover in a vectorized way, the table of parent genes is
        converted to long format. If there are four genes, then the genes of
        parent1 are stored in elements 0-3, the genes of parent2 in elements
        4-7. Child genes are initially stored as the index of the parent genes
        rather than the values of the parent genes and they are assumed initially
        to take on only the value of parent1's genes. So child1's genes are set
        initially to [0,1,2,3], child2's genes to [8,9,10,11], child3's to
        [16,17,18,19] and so on all stored in a single 1d numpy array. Then each
        gene has a 50% chance of having 4 added to it, which effectively switches
        the gene from referencing parent1 to parent2. E.g. if the second gene of
        child1 is switched, then child1's genes are now [0,5,2,3]. If the third
        gene of child2 is switched, then his genes are now [8,9,14,11]. All of
        the children genes are stored in a large list ([0,5,2,3,8,9,14,11...])
        which is used as an index to retrieve the actual genes held by the
        parents.

        Mutation works somewhat similarly. A boolean array is randomly generated
        with each element having a p_mutate probability of being True. Another
        array of the same length containing uniform random values between -20
        and +20 (totally arbitrary choice on my part that I'll have to fix) is
        also created. Multiply the two array so that most value are zero and
        add the resulting array to the child genes. This means each child gene
        has a p_mutate probability of having a random number added to it.


        Returns
        -------
        children: a numpy array of shape (num_children, num_genes).
        '''
        #Calculate crossover
        n, m = parents.shape
        a = np.tile(np.array(range(m)), n/2)
        b = np.array(range(0, n*m, m*2)).repeat(m)
        #the random element that selects genes from parent A or B
        shift = np.random.choice([0,m], m*n/2)
        index = a + b + shift
        parents_ = parents.reshape(n*m)
        children = parents_[index]


        #mutation
        mutate = np.random.choice([0,1], n*m/2, p = [1-self.p_mutate, self.p_mutate])
        mut_value = np.random.uniform(-20,20, n*m/2)
        children += mutate*mut_value
        children = children.reshape(n/2, m)
        return children

    def select_survivors(self, fitness_scores, k, n):
        '''
        Just like the tournament selection in select_parents except it chooses
        the worst members to kill off. Randomly selects k members of the
        population and "kills" the worst one. Repeats this process until n
        members of the population are removed.

        Parameters
        ----------
        fitness_scores: list
            A list of fitness scores of the population. Lower scores are better.
        k: int
            The number of members of the population to enter the tournament.
        n: int
            The number of members to remove from the population. Same as the
            number of tournaments to run.

        Returns
        -------
        living_index: a list containing the index of those who survived.
        '''
        fitness_copy = fitness_scores[:]
        living = set(range(len(fitness_scores)))
        dead = set()
        while len(dead) < n:
            potentials = sample(fitness_copy, k)
            worst = max(potentials)
            index = fitness_scores.index(worst)
            while index in dead:
                index = fitness_scores[index+1:].index(worst) + index + 1
            dead.add(index)
            fitness_copy.remove(worst)
        living = living.difference(dead)
        return list(living)

    def report(self):
        '''
        Creates a series of plots so the user can see how the population evolved
        over time.
        '''
        import matplotlib.pyplot as plt
        pd.Series(self.best_score).plot(title = 'best score')
        plt.show()
        pd.Series(self.avg_score).plot(title = 'avg score')
        plt.show()
        i = 0
        for p in self.pops:
            p = pd.DataFrame(p)
            pd.scatter_matrix(p)
            plt.suptitle('Population at iteration %s' %i)
            plt.show()
            i += 10
        p = pd.DataFrame(self.pop)
        pd.scatter_matrix(p)
        plt.suptitle('Final population.')
        plt.show()


if __name__ == '__main__':
    from time import time
    start = time()
    #Create some fake regression data.
    X = pd.DataFrame(np.random.randn(5000,4), columns = ['x1','x2','x3','x4'])
    params = [.25, 5,-12,0]
    err = np.random.randn(5000)
    y = X.dot(params) + X['x2']**2 + err*4
    s2 = time()
    #Create the regressor
    reg = GARegressor(fit_intercept = False, iterations = 100, p_mutate = .10)
    reg.fit(X, y)
    f = time()
    #Report training time and accuracy statistics.
    print "Read time: ", s2 - start
    print "Fit time: ", f - s2
    yhat = reg.predict(X)
    print "R2 score: ", metrics.r2_score(y, yhat)
    print "RMSE: ", metrics.mean_squared_error(y, yhat)**0.5
    reg.report()
