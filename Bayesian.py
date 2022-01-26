import numpy as np

# def surrogate_belief(x,f,x_star,theta):
#     return mu_star, varSigma_star


def f(x, A=1, B=0, C=0):
    return A*(6*x-2)**2*np.sin(12*x-4) + B*(x-0.5) + C

# remove points from an array
x_2 = np.arange(10)
index = np.random.permutation(10)
x_1 = x_2[index[0:3]]
x_2 = np.delete(x_2, index[0:3])
# remove largest element
ind = np.argmax(x_2)
x_1 = np.append(x_1, x_2[ind])
x_2 = np.delete(x_2, ind)
