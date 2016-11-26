"""
Implementation of a general Variational Bayes class
Author: Jake Soloff
"""

import numpy as np

class VB(object):
	def __init__(self, seed=0, maxLaps=1000, tol=1e-8): 
		self.seed      = seed
		self.PRNG      = np.random.RandomState(self.seed)
		self.ELBO      = []
		self.maxLaps   = maxLaps
		self.tol       = tol
		self.converged = False

	def run(self, model, Data, LP=None, SS=None):
		prevELBO    = -float('Inf')
		isConverged = False

		for lap in xrange(1, self.maxLaps + 1):
			# Local step (E - step)
			LP = model.calcLP(Data, LP, SS)  

			# Summary calculations
			SS = model.calcSS(Data, LP, SS)  

			# Global step (M - step)
			model.updatePost(SS)         

			# ELBO calculation
			ELBO = model.calcELBO(Data, SS, LP)
			self.ELBO.append(ELBO[0])

			if (np.abs(prevELBO - ELBO) < self.tol):
				self.isConverged = True
				break
			prevELBO = ELBO
		return self.ELBO, self.converged, model.Post, LP
