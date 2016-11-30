"""
Abstract Model class. All models should inherit from this class
Author: Eli Ben-Michael
"""


class AbstractModel(object):

    def __init__(self):
        """Constructor"""

    def assign_data(self, data):
        """Giving data to a model"""
        raise NotImplementedError

    def init_v_params(self):
        """Initialize variational parameters"""
        raise NotImplementedError

    def calc_LP(self):
        """Update the local parameters of the the model"""
        for node_name in self.local_parameters:
            self.nodes[node_name].vi_update()

    def calc_SS(self):
        """Compute the sufficient statistics for the global parameters"""
        raise NotImplementedError

    def calc_GP(self):
        """Update the global parameters of the model"""
        ss = self.calc_SS()
        for node_name in self.global_parameters:
            self.nodes[node_name].vi_update(ss)

    def calc_elbo(self):
        """Compute the current ELBO"""
        elbo = 0
        # compute node elbos
        for node in self.nodes.values():
            elbo += node.calc_elbo()
        # compute data elbo
        elbo += self.calc_data_elbo()
        return(elbo)

    def calc_data_elbo(self):
        """Compute ELBO conrtibution from likelihood"""
        raise NotImplementedError
