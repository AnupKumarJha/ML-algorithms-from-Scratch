import numpy as np


def linear_kernel(**kwargs):
    """Linear Kernel is used when the data is Linearly separable, that is,
    it can be separated using a single Line. It is one of the most common
    kernels to be used. It is mostly used when there are a Large number of
    Features in a particular Data Set.
                    (X^T)Y
    """

    def fun(x, y):
        return np.inner(x, y)

    return fun


def polynomail_kernel(power, coef, **kwargs):
    """The polynomial kernel is non-stationary kernel.
    Polynomial kernel is well suited for problems where all
    the training data is normalized.
                ((alpha)*(x^T)+coef)^power
    """

    def fun(x, y):
        return (np.inner(x, y) + coef) ** power

    return fun


def rbf_kernel(gamma, **kwargs):
    """It is general-purpose gaussian kernel; used when there is
    no prior knowledge about data
                        exp(-(gamma)||x-y||^2)
    """

    def fun(x, y):
        distance = np.linalg.norm(x - y) ** 2
        return np.exp(-gamma * distance)

    return fun


def laplace_rbf_kernel(sigma, **kwargs):
    """Equivalent to gaussian(rbf) kernel except it less
     sensitive to changes in sigma parameter
                exp(-1/(sigma)||x-y||)
     """

    def fun(x, y):
        distance = np.abs(np.linalg.norm(x - y))
        return np.exp(-1 / sigma * distance)

    return fun


def sigmoid_kernel(**kwargs):
    """The Hyperbolic Tangent kernel is also known as sigmoid kernel
    and used in Multilayer Perceptron(MLP) kernel used as activation function
                    tanh((gamma)*((x^T)y)+c)
    Common value for alpha is 1/N where N=No of dimension
    more ref::http://www.csie.ntu.edu.tw/~cjlin/papers/tanh.pdf
    """

    def fun(x, y):
        alpha = 1.0 / x.shape[1]
        return np.tanh(alpha * np.inner(x, y))
