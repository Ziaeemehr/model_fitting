'''
Reference:
    https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
'''

import numpy as np
import pylab as plt
from math import exp
from time import time
from numpy import arange
from matplotlib import pyplot
from numpy.random import randn, rand


# objective function
def objective(x):
    return x[0]**2.0
    # return np.sin(x[0])


def simulated_annealing(objective, bounds, n_iterations,
                        step_size, temp):
    '''
    simulated annealing algorithm
    '''
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # evaluate the initial point
    best_eval = objective(best)

    # current working solution
    curr, curr_eval = best, best_eval

    scores = list()
    # run the algorithm
    for i in range(n_iterations):

        # take a step
        candidate = curr + randn(len(bounds)) * step_size

        # evaluate candidate point
        candidate_eval = objective(candidate)

        # check for new best solution
        if candidate_eval < best_eval:

            # store new best point
            best, best_eval = candidate, candidate_eval

            # keep track of scores
            scores.append(best_eval)

            # report progress
            print(
                '->{:5d} f({:9.5f}) = {:10.5f}'.format(i, best[0], best_eval))
            # print(i, best, best_eval)

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate temperature for current epoch
        t = temp / float(i + 1)

        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)

        # check if we should keep the new point
        if (diff < 0) or (rand() < metropolis):
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

    return [best, best_eval, scores]


if __name__ == "__main__":

    # convex unimodal optimization function

    num_panels = 4
    fig, ax = plt.subplots(ncols=num_panels, figsize=(12, 3.5))
    np.random.seed(1)

    # plot the function that we want to find the minimum ----------------------
    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    inputs = np.arange(r_min, r_max, 0.1)
    # compute targets
    results = [objective([x]) for x in inputs]
    # create a line plot of input vs result
    ax[0].plot(inputs, results)
    # define optimal input value
    x_optima = 0.0
    # draw a vertical line at the optimal input
    ax[0].axvline(x=x_optima, ls='--', color='red')
    # show the plot
    # plt.show()

    # explore temperature vs algorithm iteration for simulated annealing ------
    # total iterations of algorithm
    iterations = 100
    # initial temperature
    initial_temp = 10
    # array of iterations from 0 to iterations - 1
    iterations = [i for i in range(iterations)]
    # temperatures for each iterations
    temperatures = [initial_temp/float(i + 1) for i in iterations]
    # plot iterations vs temperatures
    ax[1].plot(iterations, temperatures)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Temperature')
    plt.tight_layout()

    # explore metropolis acceptance criterion for simulated annealing ---------
    # metropolis acceptance criterion
    differences = [0.01, 0.1, 1.0]
    for d in differences:
        metropolis = [exp(-d/t) for t in temperatures]
        # plot iterations vs metropolis
        label = 'diff=%.2f' % d
        ax[2].plot(iterations, metropolis, label=label)
    # inalize plot
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Metropolis Criterion')
    ax[2].legend()
    

    # find the global minimum using optimization ------------------------------

    # define the total iterations
    n_iterations = 1000
    # define the maximum step size
    step_size = 0.1
    # initial temperature
    temp = 10
    # define range for input
    bounds = np.asarray([[-5.0, 5.0]])

    start_time = time()
    # perform the simulated annealing search
    best, score, scores = simulated_annealing(objective,
                                              bounds,
                                              n_iterations,
                                              step_size,
                                              temp)
    print('Done in {:g} seconds'.format(time() - start_time))
    print('f(%s) = %f' % (best, score))
    ax[3].plot(scores, '.-')
    ax[3].set_xlabel('Improvement Number')
    ax[3].set_ylabel('Evaluation f(x)')

    for i in range(num_panels):
        ax[i].margins(x=0.01, y=0.01)

    plt.savefig("fig.png", dpi=150)
    plt.close()
