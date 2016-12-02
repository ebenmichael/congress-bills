"""
Implementation of Latent Community Ideal Point Model
Author: Eli Ben-Michael
"""

import numpy as np
from .idealpoint import IdealPointModel
from .stochasticblock import StochasticBlockModel
from .modelnodes import AbstractNode


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

    def get_v_params(self):
        """Return the variational parameters"""
        return(self.v_mean, self.v_var)

    def init_v_params(self):
        """Initialize the variational parameters"""
        # initialize from prior distribution
        self.v_mean = np.empty((self.n_communities, self.dim))
        for i in range(self.n_communities):
            self.v_mean[i, :] = self.prior_mean[i, :] + \
                                self.prior_var * np.random.randn(2)
        self.v_var = self.prior_var

    def vi_update(self):
        """Update variational parameters"""
        raise NotImplementedError

    def calc_elbo(self):
        """Compute the contribution to elbo from prior/entropy"""
        raise NotImplementedError


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

        # add a new node for the community means
        self.cmean_node = CommunityMeanNode()

        # initialize Ideal Point Model
        IdealPointModel.__init__(self, dim, **kwargs)
        # initialize stochastic block model
        StochasticBlockModel.__init__(self, n_communities)

        # point ip_node and resp_node to the community means
        self.ip_node.cmean_node = self.cmean_node
        self.resp_node.cmean_node = self.cmean_node

        # tell ideal point and responsibilities to use new updates
        self.ip_node.set_lcipm(True)
        self.resp_node.set_lcipm(True)

        # update the dicts and list that keep track of nodes
        self.nodes = {"ideal_point": self.ip_node,
                      "discrimination": self.disc_node,
                      "difficulty": self.diff_node,
                      "responsibility": self.resp_node,
                      "coexpression": self.coexp_node,
                      "cluster_probability": self.cprob_node,
                      "cluster_mean": self.cmean_node}
        # keep track of user vs document parameters
        self.doc_nodes = ["discrimination", "difficulty"]
        self.user_nodes = ["ideal_point"]
        self.local_parameters = ["ideal_point", "discrimination",
                                 "difficulty", "responsibility"]
        self.global_parameters = ["coexpression",
                                  "cluster_probability",
                                  "cluster_mean"]

    def assign_data(self, data):
        """Assign data to the nodes
        Args:
            data: tuple(ndarray), first element is interaction data,
                  second element is caucus co-membership data
        """
        # assign data for ideal point model
        IdealPointModel.assign_data(self, data[0])
        # assign data for stochastic block model
        StochasticBlockModel.assign_data(self, data[1])

    def init_v_params(self):
        """Initialize variational parameters"""
        # initialize parameters for ipm and sbm
        IdealPointModel.init_v_params(self)
        StochasticBlockModel.init_v_params(self)
        # initialize cluster means
        self.cmean_node.init_v_params(self.n_items, self.n_communities)

    def calc_data_elbo(self):
        """Add the ELBOs from the likelihood terms of the ipm and sbm"""
        elbo1 = IdealPointModel.calc_data_elbo(self)
        elbo2 = StochasticBlockModel.calc_data_elbo(self)
        return(elbo1 + elbo2)

    def calc_SS(self):
        """Compute the sufficient statistics"""
        return(StochasticBlockModel.calc_SS(self))
