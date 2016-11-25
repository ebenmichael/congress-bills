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

    def get_nodes(self):
        """Return a dictionary of (node_name, node)"""
        return(self.nodes)
