# https://lmfit.github.io/lmfit-py/intro.html
import lmfit
import numpy as np
import pylab as plt
from numpy import sin, exp
from numpy import linspace, random
from scipy.optimize import leastsq
from lmfit import minimize, Parameters


def model(amp, freq, phaseshift, decay):
    return amp * sin(x*freq + phaseshift) * exp(-x*x*decay)


def residual(variables, x, data, eps_data):
    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    phaseshift = variables[1]
    freq = variables[2]
    decay = variables[3]

    y_model = model(amp, freq, phaseshift, decay)

    return (data-y_model) / eps_data


# generate synthetic data with noise
x = linspace(0, 100)
eps_data = random.normal(size=x.size, scale=0.2)
data = model(7.5, 0.22, 2.5, 0.01) + eps_data
# data = 7.5 * sin(x*0.22 + 2.5) * exp(-x*x*0.01) + eps_data

variables = [10.0, 0.2, 3.0, 0.007]
p = leastsq(residual, variables, args=(x, data, eps_data))[0]
print(p)
# plt.plot(x, data, c="b", label="data")
# plt.plot(x, model(*p), c="r", label="fit")
# plt.legend()
# plt.show()
# exit(0)

# different implementation


def residual(params, x, data, eps_data):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)
    return (data-model) / eps_data


params = Parameters()
params.add('amp', value=10)
params.add('decay', value=0.007)
params.add('phase', value=0.2)
params.add('frequency', value=3.0)

# params = Parameters()
# params.add('amp', value=10, vary=False)
# params.add('decay', value=0.007, min=0.0)
# params.add('phase', value=0.2)
# params.add('frequency', value=3.0, max=10)

# params['amp'].vary = False
# params['decay'].min = 0.10


methods = ['leastsq', 'nelder', 'lbfgsb']

for m in methods:

    print(m)
    p = minimize(residual, params,
                 args=(x, data, eps_data),
                 method=m)
    # lmfit.printfuncs.report_fit(p.params, min_correl=0.5)
    plt.plot(x, residual(p.params, x, data, eps_data)+data,
             lw=1,
             label=m)
plt.plot(x, data, lw=2, label='data', color='k')
plt.legend(frameon=False)
plt.show()
