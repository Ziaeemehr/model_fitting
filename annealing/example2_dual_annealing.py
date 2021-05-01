import numpy as np
import pylab as plt
from math import exp
from time import time
from numpy import arange
from matplotlib import pyplot
from numpy.random import randn, rand
from scipy.optimize import dual_annealing


def func(x):
    return np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)


def objective(x):
    return (x[0]-1)**2.0


if __name__ == "__main__":

    lw = [-5.12] * 10
    up = [5.12] * 10
    result = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
    print(result.x)

    print('f(x) = {:g}'.format(func(result.x)))

    print("*"*70)
    maxiter = 1000
    bounds = np.asarray([[-5.0, 5.0]])
    result = dual_annealing(objective,
                            bounds=bounds,
                            seed=1234,
                            maxiter=maxiter)
    print(result.x)
    print('f(x) = {:g}'.format(objective(result.x)))
