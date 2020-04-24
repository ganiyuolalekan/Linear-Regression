########################################################################################
# Code By Ganiyu Olalekan
########################################################################################

import numpy as np
from time import time
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x, y, alpha=0.01, num_iter=1000, verbose=False):
        assert (
            type(x) == np.ndarray and type(y) == np.ndarray and (len(x.shape) >= 1 and len(y.shape) == 1)
        ), "Their are issues with the provided data-set"

        self.alpha = alpha
        self.verbose = verbose
        self.num_iter = num_iter
        self.X = x.reshape(x.shape[0], 1)
        self.y = y.reshape(y.shape[0], 1)
        self.theta = np.zeros((1, self.X.shape[1] + 1))

    # Public Methods

    def fit(self, timeit=True, count_at=100):
        start = time()
        count = count_at
        self.__add_intercept()

        for _ in range(self.num_iter):
            h = self.__hypothesis()
            cost = self.__cost(h)

            theta0 = self.theta[0, 0] - (self.alpha * (h - self.y).mean())
            theta1 = self.theta[0, 1] - (self.alpha * ((h - self.y) * self.X[:, 1]).mean())

            self.theta = np.array([[theta0, theta1]])

            if self.verbose and count_at == count:
                count = 0
                print(f"cost {(_ + 1)}: {cost}, theta: {self.theta}")

            count += 1

        if timeit:
            print(f"Ran in {round(time() - start, 2)}secs")

    def plot_regression_graph(self):
        x = self.X[:, 1]
        plt.plot(x, self.y, '.', x, self.__hypothesis(), '-')
        plt.show()
    ########################################################################################

    # private Methods

    def __add_intercept(self):
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)

    def __cost(self, h):
        return ((h - self.y) ** 2).mean() / 2

    def __hypothesis(self):
        return np.dot(self.X, self.theta.T)
    ########################################################################################


# Through Wisdom is a house built,
# by Understanding it is established and
# by Knowledge all corners and rooms are
# filled with all manner of pleasant riches
# and treasures
#
# Ref Proverbs 3: 19 - 20
