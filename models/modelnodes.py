"""
Collection of generic parameter node classes
"""
import numpy as np
import math_utils
from scipy import optimize
from collections import defaultdict
from functools import partial
import joblib
import multiprocessing


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

    def init_v_params(self, n_items):
        """Initialize the naive mean field variational parameters
        Args:
            n_items: int, total number of items (bills/documents/users/etc.)
        """
        self.n_items = n_items
        self.v_mean = np.random.randn(n_items, self.dim)
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
        """for bill in self.data.keys():
            self.update_v_mean(bill)
        bills = [bill for bill in self.data.keys()]"""
        n_jobs = multiprocessing.cpu_count()
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.update_v_mean)(bill)
                                       for bill in self.data.keys())
        start = self.v_mean.reshape(self.n_items * self.dim)
        """output = optimize.minimize(self.objective, start, jac=self.gradient,
                                   method="BFGS")
        print(output)
        out = output["x"]
        out = math_utils.gradient_descent(start, self.gradient)
        self.v_mean = out.reshape(self.n_items, self.dim)"""

    def update_v_mean(self, bill):
        """Update the mean of the variational distribution for bill"""
        # optimize starting at previous value
        if bill % 10 == 0:
            print("updating bill %i"%bill)
        start = self.v_mean[bill, :]
        # partial application of gradient
        grad = partial(self.grad_for_bill, bill)
        objective = partial(self.objective_for_bill, bill)
        """output = optimize.minimize(objective, start, jac=grad,
                                   method="BFGS")
        print(output)
        self.v_mean[bill, :] = output["x"]"""
        out = math_utils.gradient_descent(start, grad)
        # print(out)
        self.v_mean[bill, :] = out

    def update_v_var(self):
        """Update the variance of the variational distribution"""
        num = self.n_items * self.dim
        s1 = num / self.prior_var
        s2 = 0
        for i, interaction in enumerate(self.data):
            s2 += self.get_update_from_point(interaction)
            if(i % 10000 == 0):
                print(i)
        # update the variational variance
        self.v_var = num / (s1 + s2)

    def grad_for_bill(self, bill, value):
        raise NotImplementedError

    def gradient(self, value):
        raise NotImplementedError

    def objective_for_bill(self, bill, value):
        raise NotImplementedError

    def objective(self, value):
        raise NotImplementedError
