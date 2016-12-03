"""
Implementation of a general Variational Bayes class
Author: Jake Soloff (edited by Eli Ben-Michael >:))
"""

import numpy as np


class VB(object):
    def __init__(self, seed=0, maxLaps=1000, tol=1e-8):
        self.seed = seed
        self.PRNG = np.random.RandomState(self.seed)
        self.ELBO = []
        self.maxLaps = maxLaps
        self.tol = tol
        self.isConverged = False

    def run(self, model, data):
        """Run variational inference. Updates model in place
        Args:
            model: AbstractModel, model to update in place
            data: ndarray, tuple of ndarrays, data to fit model on
        Returns:
            ELBO: list, trace of ELBO values
            isConverged: bool, whether the ELBO converged
        """
        # assign the data and initialize variational parameters
        model.assign_data(data)
        model.init_v_params()
        prevELBO = -float('Inf')
        lap = 0
        while not self.isConverged and lap < self.maxLaps:
            print("Computing Local Parameters")
            # Local step (E - step)
            model.calc_LP()

            print("Computing Global Parameters")
            # Global step (M - step)
            model.calc_GP()

            # ELBO calculation
            elbo = model.calc_elbo()

            self.ELBO.append(elbo)
            self.isConverged = np.abs(prevELBO - elbo) < self.tol
            lap += 1
            prevELBO = elbo
            print(lap, elbo)
        return(self.ELBO, self.isConverged)
