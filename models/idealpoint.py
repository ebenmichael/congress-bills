"""
Implementation of a Bayesian Ideal Point Model
Author: Eli Ben-Michael
"""

import numpy as np
import math_utils
from scipy import optimize
from collections import defaultdict
from functools import partial
import joblib
import multiprocessing
from modelnodes import GaussianNode
from abstractmodel import AbstractModel


class DiscNode(GaussianNode):
    """Discrimination parameters in IPM"""

    def assign_params(self, diff_node, ip_node):
        """Point to the difficulty node and the ideal point node"""
        self.diff_node = diff_node
        self.ip_node = ip_node

    def get_update_from_point(self, interaction):
        """Compute the contribution of this interaction to the variance update
        Args:
            interaction: ndarray, length 3, (bill, person, action)
        Returns:
            update_contrib: float, the contribution
        """
        bill, person, action = interaction
        disc_v_mean = self.v_mean[bill, :]
        diff_v_mean = self.diff_node.v_mean[bill, :]
        diff_v_var = self.diff_node.v_var
        ip_v_mean = self.ip_node.v_mean[person, :]
        ip_v_var = self.ip_node.v_var
        param_val = np.dot(disc_v_mean, ip_v_mean - diff_v_mean)
        sig_prime = math_utils.sigmoid_prime(param_val)
        s1 = self.dim * (diff_v_var + ip_v_var)
        s2 = np.linalg.norm(ip_v_mean - diff_v_mean) ** 2

        return(sig_prime * (s1 + s2))

    def gradient(self, value):
        """Compute the gradient of the ELBO with respect to the variational mean
        Args:
            value: ndarray, length(dim), value to compute gradient at
        Returns:
            grad: ndarray, length(dim), the value of the gradient
        """
        # print("Gradient computed")
        # prior_deriv = value - self.prior_mean
        # prior_deriv /= self.prior_var

        grad = np.zeros(self.n_items * self.dim)
        for bill in self.data.keys():
            bill_value = value[bill: bill + self.dim]
            bill_grad = self.grad_for_bill(bill, bill_value)
            grad[2 * bill: 2 * bill + self.dim] = bill_grad
        # print(grad)
        return(grad)

    def grad_for_bill(self, bill, value):
        """Compute the gradient of the ELBO for a given bill
        Args:
            bill: int, bill to compute gradient for
            value: ndarray, length dim, current value of v mean
        Returns:
            curr_grad: ndarray, length n_items * dim, updated gradient
        """
        prior_deriv = value - self.prior_mean
        prior_deriv /= self.prior_var
        # get bill parameters
        v_mean = value.reshape(1, len(value))
        disc_v_var = self.v_var
        diff_v_mean = self.diff_node.v_mean[bill, :]
        diff_v_var = self.diff_node.v_var
        # get person parameters for each vote
        people = self.data[bill][:, 1]
        ip_v_means = self.ip_node.v_mean[people, :]
        ip_v_var = self.ip_node.v_var
        # compute the errors
        votes = self.data[bill][:, 2]
        votes = votes.reshape(len(votes), 1)
        param_vals = np.dot(ip_v_means - diff_v_mean, v_mean.T)
        probs = math_utils.sigmoid(param_vals)
        errors = (votes - probs)
        diff_params = ip_v_means - diff_v_mean

        s1 = 0.5 * math_utils.sigmoid_double_prime(param_vals)
        s1 = s1.reshape(len(votes), 1) * diff_params
        p1 = 0  # self.dim * disc_v_var * (ip_v_var + diff_v_var)
        p1 += disc_v_var * np.sum(diff_params ** 2, axis=1)
        p1 += (ip_v_var + diff_v_var) * np.linalg.norm(v_mean)
        p1 = p1.reshape(len(votes), 1)
        s1 *= p1

        s2 = math_utils.sigmoid_prime(param_vals) * v_mean
        s2 *= (diff_v_var + ip_v_var)
        #print(errors)
        #print(diff_params)
        #print(s1)
        #print(s2)
        total = errors * diff_params - s1 - s2
        grad = total.sum(0)
        return(-(prior_deriv + grad))

    def objective(self, value):
        """Compute the portion of the ELBO which depends on these nodes"""
        elbo = sum(self.objective_for_bill(bill,
                                           value[2*bill: 2*bill + self.dim])
                   for bill in self.data.keys())
        print(elbo)
        return(elbo)

    def objective_for_bill(self, bill, value):
        """Compute the portion of the ELBO which depends on the variational
           mean for bill
        Args:
            bill: int, bill to compute gradient for
            value: ndarray, length dim, current value of v mean
        Returns:
            obj: float, the portion of the elbo which depends on the bill
        """
        # compute elbo contribution from parameters
        mean_diff = value - self.prior_mean
        elbo1 = (self.v_var * self.dim + np.sum(mean_diff ** 2))
        elbo1 /= 2 * self.prior_var

        # compute elbo contribution from data
        # get bill parameters
        v_mean = value
        disc_v_var = self.v_var
        diff_v_mean = self.diff_node.v_mean[bill, :]
        diff_v_var = self.diff_node.v_var
        # get person parameters for each vote
        people = self.data[bill][:, 1]
        ip_v_means = self.ip_node.v_mean[people, :]
        ip_v_var = self.ip_node.v_var
        votes = self.data[bill][:, 2]
        param_vals = np.dot(ip_v_means - diff_v_mean, v_mean)
        diff_params = ip_v_means - diff_v_mean

        s1 = votes * param_vals - np.log(1 + np.exp(param_vals))

        s2 = 0.5 * math_utils.sigmoid_prime(param_vals)
        p1 = 0  # self.dim * disc_v_var * (ip_v_var + diff_v_var)
        p1 += disc_v_var * np.sum(diff_params ** 2, axis=1)
        p1 += (ip_v_var + diff_v_var) * np.sum(value ** 2)
        s2 *= p1

        elbo2 = (s1 - s2).sum()
        return(-(elbo1 + elbo2))


class DiffNode(GaussianNode):
    """Difficulty parameters in IPM"""

    def assign_params(self, disc_node, ip_node):
        """Point to the discrimination node and the ideal point node"""
        self.disc_node = disc_node
        self.ip_node = ip_node


class IdealPointNode(GaussianNode):
    """Ideal point parameters in IPM"""

    def assign_params(self, disc_node, diff_node):
        """Point to the discrimination and difficulty nodes"""
        self.disc_node = disc_node
        self.diff_node = diff_node


class IdealPointModel(AbstractModel):
    """Bayesian Ideal Point Model"""

    def __init__(self, dim, disc_prior_mean=None, disc_prior_var=None,
                 diff_prior_mean=None, diff_prior_var=None,
                 ip_prior_mean=None, ip_prior_var=None):
        """Constructor
        Args:
            dim: int, dimension of discrimination parameters
            disc_prior_mean: ndarray, prior mean of discrim, defaults to 0
            disc_prior_var: float, prior variance of discrim, defaults to 1
            diff_prior_mean: ndarray, prior mean of difficulty, defaults to 0
            diff_prior_var: float, prior variance of difficulty, defaults to 1
            ip_prior_mean: ndarray, prior mean of ideal point, defaults to 0
            ip_prior_var: float, prior variance of ideal point, defaults to 1
        """

        self.dim = dim
        self.disc_node = DiscNode(dim, disc_prior_mean, disc_prior_var)
        self.diff_node = DiffNode(dim, diff_prior_mean, diff_prior_var)
        self.ip_node = IdealPointNode(dim, ip_prior_mean, ip_prior_var)

        # point the nodes to the other nodes
        self.disc_node.assign_params(self.diff_node, self.ip_node)
        self.diff_node.assign_params(self.disc_node, self.ip_node)
        self.ip_node.assign_params(self.disc_node, self.diff_node)

        self.nodes = {"discrimination": self.disc_node,
                      "difficulty": self.diff_node,
                      "ideal_point": self.ip_node}
        # keep track of user vs document parameters
        self.doc_nodes = ["discrimination", "difficulty"]
        self.user_nodes = ["ideal_point"]

    def assign_data(self, data):
        """Give the model data
        Args:
            data: ndarray, shape (n_interactions, 3), interaction data"""
        data = np.array(data, dtype=int)
        self.data = data
        # seperate by bill and person
        bill_data = defaultdict(list)
        user_data = defaultdict(list)
        for i in range(data.shape[0]):
            bill_data[data[i, 0]].append(data[i, :])
            user_data[data[i, 1]].append(data[i, :])
        # convert to numpy arrays
        bill_data = {i: np.array(bill_data[i]) for i in bill_data.keys()}
        user_data = {i: np.array(user_data[i]) for i in user_data.keys()}
        for node in self.nodes.keys():
            if node in self.doc_nodes:
                self.nodes[node].assign_data(bill_data)
            elif node in self.user_nodes:
                self.nodes[node].assign_data(user_data)

    def init_v_params(self):
        """Initialize variational parameters"""
        n_unique_items = len(np.unique(self.data[:, 0]))
        n_unique_users = len(np.unique(self.data[:, 1]))

        self.disc_node.init_v_params(n_unique_items)
        self.diff_node.init_v_params(n_unique_items)
        self.ip_node.init_v_params(n_unique_users)
