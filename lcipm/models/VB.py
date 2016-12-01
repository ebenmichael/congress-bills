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
                lap = 1
                converged = False
                while not converged:
                        print("Computing Local Parameters")
                        # Local step (E - step)
                        model.calc_LP()
                        # Global step (M - step)
                        model.calc_GP()
                        # ELBO calculation
                        elbo = model.calc_elbo()
                        self.ELBO.append(elbo)
                        if np.abs(prevELBO - elbo) < self.tol:
                                self.isConverged = True
                                converged = True
                        if lap >= self.maxLaps:
                                converged = True
                                self.isConverged = False
                        print(lap, "ELBO", elbo)
                        lap += 1
                        prevELBO = elbo
                return(self.ELBO, self.isConverged)
