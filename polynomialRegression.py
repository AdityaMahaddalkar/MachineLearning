import numpy as np


class PolynomialRegressor():
    def __init__(self):
        self.weights = None

    def fit(self, X, y, learning_rate=0.0000000005, iterations=10000):
        '''
        Input: X is one dimensional array of samples of one feature
               y is one dimensional array of output corresponding to features
               learning_rate is default 0.05
               iterations are default 10000
        Execution: Computes upto X^4 validations of samples and outputs the most probable sample
        Increment: Try implementing R square function
        '''

        try:
            if X.size != y.size:
                raise Exception("Number of features and outputs not equal")
        except Exception as e:
            print(e)
            exit()

        m = X.size
        X.shape = (m, 1)

        # Initialize weights

        self.weights = np.random.random((5, 1))
        # Create columns of bias, X, X**2, X**3, X**4
        try:
            X = np.concatenate(
                (np.ones(shape=(m, 1), dtype=X.dtype), X, X**2, X**3, X**4), axis=1)
            print(X.shape)
        except Exception as e:
            print(e)

        # Computing loss function
        #hypothesis = np.dot(X, self.weights)
        #loss = hypothesis - y
        #cost = np.sum(loss**2)/m

        for i in range(iterations):
            hypothesis = np.dot(X, self.weights)
            loss = hypothesis - y
            cost = np.sum(loss**2)/m
            # How to calculate del J / del weights ???
            self.weights = self.weights - learning_rate * del_J

            if i % 100 == 0:
                print('At iteration {} cost is {}'.format(i, cost))

        print('Finally the weights are')
        print(self.weights.shape)

    def predict(self, X):
        '''
        Input: X is one dimensional numpy array
        '''
        m = X.size
        X.shape = (m, 1)

        # Create columns of bias, X, X**2, X**3, X**4
        try:
            X = np.concatenate(
                (np.ones(shape=(m, 1), dtype=X.dtype), X, X**2, X**3, X**4), axis=1)
        except Exception as e:
            print(e)

        return np.dot(X, self.weights)


poly_reg = PolynomialRegressor()
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
poly_reg.fit(X, y)
print(poly_reg.predict(np.array([1, 2, 3, 4, 5])))
