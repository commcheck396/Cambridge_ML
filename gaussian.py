import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
def rbf_kernel(x1, x2, varSigma, lengthscale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma*np.exp(-np.power(d, 2)/lengthscale)
    return K

def lin_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*x1.dot(x1.T)
    else:
        return varSigma*x1.dot(x2.T)
        
def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])


def periodic_kernel(x1, x2, varSigma, period, lenthscale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lenthscale**2)


x = np.linspace(-5, 5, 200).reshape(-1, 1)
c=x.shape
# compute covariance matrix
K = rbf_kernel(x, None, 1.0, 2.0)
# create mean vector
mu = np.zeros(c[0])
print(mu)
print(x)
# draw samples 20 from Gaussian distribution
f = np.random.multivariate_normal(mu, K, 20)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, f.T)
plt.show()

