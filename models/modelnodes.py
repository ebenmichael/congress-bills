"""
Collection of generic parameter node classes
"""
import numpy as np
from scipy import optimize
from functools import partial
import joblib


class GaussianNode(object):
    """Abstract node class for a node with isotropic guassian variational
       distribution"""

    def __init__(self, dim, prior_mean=None, prior_var=None):
        """Constructor
        Args:
            dim: int, dimension of discrimination parameters
            prior_mean: ndarray, prior mean of parameters, defaults to 0
            prior_var: float, prior variance of parameters, defaults to 1
        """
        self.dim = dim
        if prior_mean is None:
            prior_mean = np.zeros(dim)
        if prior_var is None:
            prior_var = 1
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def assign_data(self, data):
        """Give the node relevant data"""
        self.data = data

    def get_v_params(self):
        """Return the variational parameters"""
        return([self.v_mean, self.v_var])

    def init_v_params(self, n_items):
        """Initialize the naive mean field variational parameters
        Args:
            n_items: int, total number of items (bills/documents/users/etc.)
        """
        self.n_items = n_items
        self.v_mean = np.random.randn(n_items, self.dim) / 10
        self.v_var = 1.0

    def get_v_mean(self):
        """Return the mean of the variational distribution"""
        return(self.v_mean)

    def vi_update(self):
        """Perform a variational inference update"""
        self.update_v_means()
        self.update_v_var()

    def update_v_means(self):
        """Update the means of the variational distribution"""
        for item in self.data.keys():
            self.update_v_mean(item)

    def update_v_mean(self, item):
        """Update the mean of the variational distribution for bill"""
        # optimize starting at previous value
        start = self.v_mean[item, :]
        # partial application of gradient
        grad = partial(self.grad_for_item, item)
        objective = partial(self.objective_for_item, item)
        output = optimize.minimize(objective, start, jac=grad,
                                   method="L-BFGS-B")
        self.v_mean[item, :] = output["x"]

    def update_v_var(self):
        """Update the variance of the variational distribution"""
        num = self.n_items * self.dim
        s1 = num / self.prior_var
        s2 = 0
        for item in self.data.keys():
            s2 += self.var_update_from_item(item)
        # update the variational variance
        self.v_var = num / (s1 + s2)

    def compute_elbo(self):
        """Compute the ELBO from the node entropy and the prior"""
        entropy = self.n_items * self.dim / 2
        entropy *= np.log(2 * np.pi * np.exp(1) * self.v_var)

        s = np.sum((self.v_mean - self.prior_mean) ** 2, 1)
        s += self.v_var * self.dim
        s = s.sum() / (2 * self.prior_var)
        return(entropy + s)

    def objective(self, value):
        """Compute the portion of the ELBO which depends on these nodes"""
        elbo = sum(self.objective_for_item(item,
                                           value[self.dim * item:
                                                 self.dim * item + self.dim])
                   for item in self.data.keys())
        return(elbo)

    def grad_for_item(self, item, value):
        raise NotImplementedError

    def gradient_for_items(self, items, value):
        """Compute the gradient of the ELBO with respect to the variational mean
           for a given list of items
        Args:
            items: list, items to compute gradient for
            value: ndarray, length(dim), value to compute gradient at
        Returns:
            grad: ndarray, length(dim), the value of the gradient
        """

        grad = np.zeros(self.n_items * self.dim)
        for item in items:
            item_value = value[item: item + self.dim]
            item_grad = self.grad_for_item(item, item_value)
            grad[self.dim * item: self.dim * item + self.dim] = item_grad
        return(grad)

    def gradient(self, value):
        """Compute the gradient of the ELBO with respect to the variational mean
        Args:
            value: ndarray, length(dim), value to compute gradient at
        Returns:
            grad: ndarray, length(dim), the value of the gradient
        """

        items = [item for item in self.data.keys()]
        return(self.gradient_for_items(items, value))
