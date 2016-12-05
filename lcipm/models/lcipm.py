"""
Implementation of Latent Community Ideal Point Model
Author: Eli Ben-Michael
"""

import numpy as np
from .idealpoint import IdealPointModel
from .sbm import StochasticBlockModel
from .modelnodes import AbstractNode
from collections import defaultdict
import os


class CommunityMeanNode(AbstractNode):
    """Community ideal point means in LCIPM"""

    def __init__(self, dim, n_communities, prior_mean=None, prior_var=None):
        """Constructor:
        Args:
            dim: int, dimension of ideal points
            n_communities: int, number of communities
            prior_mean: ndaray, shape (n_communities, dim), prior means
            prior_var: float, prior variance for community means
        """
        self.dim = dim
        self.n_communities = n_communities
        if prior_mean is None:
            prior_mean = np.zeros((n_communities, dim))
        if prior_var is None:
            prior_var = 1
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def assign_params(self, ip_node, resp_node):
        """Point to ideal point node and resposinbility node"""
        self.ip_node = ip_node
        self.resp_node = resp_node

    def get_v_params(self):
        """Return the variational parameters"""
        return(self.v_mean, self.v_var)

    def init_v_params(self):
        """Initialize the variational parameters"""
        # initialize from prior distribution
        self.v_mean = np.empty((self.n_communities, self.dim))
        for i in range(self.n_communities):
            self.v_mean[i, :] = self.prior_mean[i, :] + \
                                self.prior_var * np.random.randn(self.dim)
        self.v_var = self.prior_var

    def vi_update(self, ss):
        """Update variational parameters
        Args:
            ss: dict, holds the sufficient statistics
        """
        self.update_v_mean(ss)
        self.update_v_var()

    def update_v_mean(self, ss):
        """Update the variational means of the community means
         Args:
            ss: dict, holds the sufficient statistics
        """
        nk = ss["N_k_full"]
        n_users = self.ip_node.n_items
        for k in range(self.n_communities):
            resp_k = self.resp_node.resp[:, k].reshape(n_users, 1)
            m1 = np.sum(resp_k * self.ip_node.v_mean, axis=0)
            m1 /= self.ip_node.prior_var
            m2 = self.prior_mean[k, :] / self.prior_var
            numerator = m1 + m2
            denom = nk[k] / self.ip_node.prior_var + 1 / self.prior_var
            self.v_mean[k, :] = numerator / denom

    def update_v_var(self):
        """Update the variational variance"""
        n_users = self.ip_node.n_items
        n_communities = self.n_communities
        self.v_var = n_communities / (n_users / self.ip_node.prior_var
                                      + n_communities / self.prior_var)

    def calc_elbo(self):
        """Compute the ELBO from the node entropy and the prior"""
        entropy = self.n_communities * self.dim / 2
        entropy *= np.log(2 * np.pi * np.exp(1) * self.v_var)

        s = np.sum((self.v_mean - self.prior_mean) ** 2, 1)
        s += self.v_var * self.dim
        s = s.sum() / (2 * self.prior_var)
        return(entropy + s)


class LCIPM(IdealPointModel, StochasticBlockModel):
    """Latent Community Ideal Point Model"""

    def __init__(self, dim, n_communities, **kwargs):
        """Constructor. Calls initializations from IdealPointModel and
           StochasticBlockModel and changes IdealPointNode and
           ResponsibilityNode.
        Args:
            dim: int, dimension of ideal points
            n_communities: int, number of communities in model
            **kwargs: key word arguments for IdealPointModel specifying
                      prior parameters
        """
        self.n_communities = n_communities
        # add a new node for the community means
        self.cmean_node = CommunityMeanNode(dim, n_communities)

        # initialize Ideal Point Model
        IdealPointModel.__init__(self, dim, **kwargs)
        # initialize stochastic block model
        StochasticBlockModel.__init__(self, n_communities)

        # point ip_node and resp_node to the community means
        self.ip_node.assign_lcipm_params(self.cmean_node, self.resp_node)
        self.resp_node.assign_lcipm_params(self.cmean_node, self.ip_node)
        self.cmean_node.assign_params(self.ip_node, self.resp_node)
        
        # tell ideal point and responsibilities to use new updates
        self.ip_node.set_model("LCIPM")
        self.resp_node.set_model("LCIPM")

        # update the dicts and list that keep track of nodes
        self.nodes = {"ideal_point": self.ip_node,
                      "discrimination": self.disc_node,
                      "difficulty": self.diff_node,
                      "responsibility": self.resp_node,
                      "coexp_rates": self.rate_node,
                      "proportions": self.prop_node,
                      "cluster_mean": self.cmean_node}
        # keep track of user vs document parameters
        self.doc_nodes = ["discrimination", "difficulty"]
        self.user_nodes = ["ideal_point"]
        self.local_parameters = ["responsibility", "ideal_point",
                                 "discrimination", "difficulty"]
        self.global_parameters = ["coexp_rates",
                                  "proportions",
                                  "cluster_mean"]

    def assign_data(self, data):
        """Assign data to the nodes
        Args:
            data: tuple(ndarray), first element is interaction data,
                  second element is caucus co-membership data
        """
        # assign data for ideal point model
        interactions = np.array(data[0], dtype=int)
        self.interactions = interactions
        # seperate by bill and person
        bill_data = defaultdict(list)
        user_data = defaultdict(list)
        for i in range(interactions.shape[0]):
            bill_data[interactions[i, 0]].append(interactions[i, :])
            user_data[interactions[i, 1]].append(interactions[i, :])
        # convert to numpy arrays
        self.bill_data = {i: np.array(bill_data[i]) for i in bill_data.keys()}
        self.user_data = {i: np.array(user_data[i]) for i in user_data.keys()}
        for node in self.nodes.keys():
            if node in self.doc_nodes:
                self.nodes[node].assign_data(self.bill_data)
            elif node in self.user_nodes:
                self.nodes[node].assign_data(self.user_data)
        # assign data for stochastic block model
        self.data = data[1]
        self.resp_node.assign_data(self.data)
        self.U = data[1].shape[0]

    def init_v_params(self):
        """Initialize variational parameters"""
        # initialize parameters for ipm and sbm
        IdealPointModel.init_v_params(self)
        StochasticBlockModel.init_v_params(self)
        # initialize cluster means
        self.cmean_node.init_v_params()

    def calc_data_elbo(self):
        """Add the ELBOs from the likelihood terms of the ipm and sbm"""
        elbo1 = IdealPointModel.calc_data_elbo(self)
        elbo2 = StochasticBlockModel.calc_data_elbo(self)
        return(elbo1 + elbo2)

    def calc_SS(self):
        """Compute the sufficient statistics"""
        return(StochasticBlockModel.calc_SS(self))

    def save(self, outdir):
        """Save model parameters
        Args:
            outdir: string, outdir to write to
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        IdealPointModel.save(self, outdir)
        StochasticBlockModel.save(self, outdir)
        # community means
        np.savetxt(os.path.join(outdir, "community_mean.dat"),
                   self.cmean_node.v_mean)
        np.savetxt(os.path.join(outdir, "community_var.dat"),
                   np.array([self.cmean_node.v_var]))
