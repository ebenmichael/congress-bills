"""
Class to implement variational inference
Author: Eli Ben-Michael
"""
import numpy as np


class VBayes(object):

    def __init__(self, tol=1e-5, max_iter=50):
        """Constructor
        Args:
            tol: float, convergence tolerance, defaults to 1e-5
        """
        self.tol = 1e-5
        self.max_iter = max_iter

    def fit(self, model, data):
        """Fit variational approximation of posterior for model to data.
           Updates model in place.
        Args:
            model: AbstractModel, the model to fit
            data: array-like, the data to use
        Returns:
            elbos: list, list of elbo values for each iteration
        """
        model.assign_data(data)
        model.init_v_params()
        converged = False
        iter_num = 1
        prev_elbo = float("-inf")
        elbos = []
        while not converged:
            print(iter_num)
            # iterate over nodes in the model
            for node_name, node in sorted(model.nodes.items(),
                                          key=lambda x: x[0], reverse=True):
                print(node_name)
                node.vi_update()
            elbo = model.compute_elbo()
            print("ELBO: " + str(elbo))
            converged = np.isclose(elbo, prev_elbo)
            elbos.append(elbo)
            prev_elbo = elbo
            iter_num += 1
            if iter_num > self.max_iter:
                converged = True
        return(elbos)
