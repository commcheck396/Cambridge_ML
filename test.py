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

def periodic_kernel(x1, x2, varSigma, period, lenthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lenthScale**2)

def gp_prediction(x1, y1, xstar, lenthScale, varSigma):
    k_starX = periodic_kernel(xstar, x1, varSigma, 4, lenthScale)
    k_xx = periodic_kernel(x1, None , varSigma, 4, lenthScale)
    k_starstar = periodic_kernel(xstar, None, varSigma, 4, lenthScale)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - k_starX.dot(np.linalg.inv(k_xx)).dot(k_starX.T)
    return mu, var, xstar

lenthScale=2
varSigma=1

N = 8
x = np.linspace(-3.1,3,N)
y = np.sin(2*np.pi/x) + x*0.1 + 0.3*np.random.randn(x.shape[0])
x = np.reshape(x,(-1,1))
y = np.reshape(y,(-1,1))
x_star = np.linspace(-20, 20, 500).reshape(-1,1)
# print(x_star)

Nsamp = 1
mu_star, var_star, x_star = gp_prediction(x, y, x_star, lenthScale, varSigma)
# x_star.reshape(-1.1)
# print (mu_star)
# print("separate")
# print(var_star)
# print("separate")
# print(x_star)
mu_star=np.squeeze(mu_star)
f_star = np.random.multivariate_normal(mu_star, var_star, Nsamp)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_star, mu_star.T)
# ax.plot(x_star, var_star.T)
ax.scatter(x, y, 100, 'k', '*', zorder=2)
# plt.fill_between(x_star, f_star.T, f_star.T, facecolor='green', alpha=0.3)
plt.show()