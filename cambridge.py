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

def gp_prediction(x1, y1, xstar, lengthScale, varSigma, noise):
    k_starX = rbf_kernel(xstar,x1,varSigma,lengthScale)
    k_xx = rbf_kernel(x1, y1,  varSigma,lengthScale)
    k_starstar = rbf_kernel(xstar,None,lengthScale,varSigma,noise)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
    return mu, var, xstar

# def periodic_kernel(x1, x2, varSigma, period, lenthScale):
#     if x2 is None:
#         d = cdist(x1, x1)
#     else:
#         d = cdist(x1, x2)
#     return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lengthScale**2)

x = np.linspace(-6, 6, 200).reshape(-1, 1)
c=x.shape
# compute covariance matrix
K = rbf_kernel(x, None, 1.0, 2.0)
# create mean vector
mu = np.zeros(200)
# draw samples 20 from Gaussian distribution
f = np.random.multivariate_normal(mu, K, 20)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, f.T)
plt.show()


# N = 5
# x = np.linspace(-3.1,3,N)
# y = np.sin(2*np.pi/x) + x*0.1 + 0.3*np.random.randn(x.shape[0])
# x = np.reshape(x,(-1,1))
# y = np.reshape(y,(-1,1))
# x_star = np.linspace(-6, 6, 500)

# x1=1
# y1=1

# lengthScale=None
# varSigma=1.0
# noise=2.0

# Nsamp = 100
# mu_star, var_star, x_star = gp_prediction(x, y, x, lengthScale, varSigma, noise)
# fstar = np.random.multivariate_normal(mu_star, var_star, Nsamp)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x_star, fstar.T)
# ax.scatter(x1, y1, 200, 'k', '*', zorder=2)
# plt.show()
