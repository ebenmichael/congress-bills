"""
Implementation of a Bayesian Stochastic Block Model
Author: Jake Soloff
"""

import numpy as np
from math import log
from scipy.special import gamma, digamma
from abstractmodel import AbstractModel
from ValBag import ValBag

class StochasticBlockModel(AbstractModel):
	"""Bayesian Stochastic Block Model"""

	def __init__(self, U, C, Data=None, **kwargs):
		"""Construct the Prior"""
		self.U = U
		self.C = C
		self.createPrior(Data=Data, **kwargs)

	def createPrior(self, Data, gamma=None, lambd=None):
		self.Prior = ValBag()
		if gamma is None:
			gamma = np.array([5.]) 
		if lambd is None:
			lambd = np.array([2.,2.])
		self.Prior.setField('gamma', gamma)
		self.Prior.setField('lambd', lambd)

	def assign_data(self, data):
		return

	def init_v_params(self):
		return

	def get_nodes(self):
		return

	## calculate local parameter dictionary
	## contains posterior responsibilities 
	def calcLP(self, Data, LP=None, SS=None):
		if LP is None:
			return self.initLP(Data)
		C = self.C
		U = self.U
		Post  = self.Post
		Prior = self.Prior

		resp = LP['resp']
		resp_= np.zeros((U,C))
		m1  = np.zeros((C,C))
		m2  = np.zeros((C,C))
		for k in range(C):
			for l in range(C):
				m1[k,l] = (digamma(Post.lam0[k,l]) - log(Post.lam1[k,l])) 
				m2[k,l] = (Post.lam0[k,l] / log(Post.lam1[k,l]))
		m1 += m1.T
		m2 += m2.T
		ElogPi = E_log_pi(Post.gamma)
		for u in range(U):
			for k in range(C):
				resp_[u,k] = np.exp(ElogPi[k] + np.dot(SS.S_uk_full[u,:],m1[k,:].T) - np.dot(SS.N_k_full,m2[k,:].T) - 1)
		for u in range(U):
			resp_[u,:] /= np.sum(resp_[u,:])

		LP['resp'] = resp_
		return LP

	## initialize LP dictionary
	## completely random init
	def initLP(self, Data):
		LP = dict()
		resp = np.random.random((U,C))
		for u in range(self.U):
			resp[u,:] /= np.sum(resp[u,:])
		#resp = np.arrayp([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
		LP['resp'] = resp
		return LP

	## calculate summary statistics
	## using posterior responsibilities
	def calcSS(self, Data, LP, SS=None):
		resp = LP['resp']
		if SS is None:
			SS = ValBag(U=self.U, C=self.C)

		Nk  = np.sum(resp, axis=0)
		SS.setField('N_k_full', Nk)

		Nkl = np.outer(Nk,Nk)
		SS.setField('N_kl_full', Nkl)

		SS.setField('S_uk_full', np.dot(Data,resp))

		Skl = np.zeros((C,C))
		for k in range(C):
			for l in range(C):
				Skl[k,l] = np.sum(np.outer(resp[:,k],resp[:,l]) * Data)
		SS.setField('S_kl_full', Skl)
		H = -resp*np.log(resp)
		SS.setField('H',np.sum(H[H < float('Inf')]))
		return SS

	## calculate global parameters 
	## given summary statistics
	def updatePost(self, SS):
		if not hasattr(self, 'Post'):
			self.Post = ValBag(U=self.U,C=self.C)
		gamma, lam0, lam1 = self.calcPost(SS)
		self.Post.setField('gamma', gamma)
		self.Post.setField('lam0',  lam0)
		self.Post.setField('lam1',  lam1)

	def calcPost(self, SS):
		Prior = self.Prior 
		gamma = Prior.gamma    + SS.N_k_full
		lam0  = Prior.lambd[0] + SS.S_kl_full
		lam1  = Prior.lambd[1] + SS.N_kl_full
		return gamma, lam0, lam1

	## evaluate the evidence lower bound
	def calcELBO(self, Data, SS, LP):
		C = self.C
		elbo  = np.zeros((C,C))
		Post  = self.Post
		Prior = self.Prior
		for k in xrange(C):
			for l in xrange(C):
				elbo[k,l] += Prior.lambd[0]*log(Prior.lambd[1]) - Post.lam0[k,l]*log(Post.lam1[k,l]) \
											     - log(gamma(Prior.lambd[0]) / gamma(Post.lam0[k,l]))
				elbo[k,l] += (SS.S_kl_full[k,l] + Prior.lambd[0] - Post.lam0[k,l])* \
													  (digamma(Post.lam0[k,l]) - log(Post.lam1[k,l]))
				elbo[k,l] -= (SS.N_kl_full[k,l] + Prior.lambd[1] - Post.lam1[k,l])* \
															   (Post.lam0[k,l] / log(Post.lam1[k,l]))
		ElogPi = E_log_pi(Post.gamma) 

		# contributions to the ELBO in order: L_data, L_ent, L_local, L_global
		return np.sum(elbo) \
				+ SS.H \
				+ np.dot(ElogPi,SS.N_k_full) \
				+ log(gamma(Prior.gamma*C)) - C*log(gamma(Prior.gamma)) \
					+ np.sum(np.log(gamma(Post.gamma))) - log(gamma(np.sum(Post.gamma))) - np.dot(ElogPi,Prior.gamma-Post.gamma)

def E_log_pi(gam):
	return digamma(gam) - digamma(np.sum(gam))


from VB import VB

# generate toy dataset
U, C = 4, 2
Data  = np.array([[10., 8., 0., 0.],
				  [8., 11., 0., 0.],
				  [0., 0.,  10.,9.],
				  [0., 0.,  9.,11.]])
#Data = np.zeros((U,U))
#pi   = np.random.dirichlet([1.]*C)
#M    = np.random.choice(C,U,p=pi)
#P    = np.array([[2.,.5],[.5,2.]]) 
#for u in xrange(U):
#	for v in xrange(u):
#		Data[u,v] = np.random.poisson(P[M[u],M[v]])
#		Data[v,u] = Data[u,v]

'''
import csv
fn = '../data/caucus/membership_110.csv'
l = []
with open(fn,'r') as f:
	reader = csv.reader(f)
	for row in reader:
		l.append([int(s) for s in row[1:]])
Data = np.array(l[1:])
Data = np.dot(Data,Data.T)

U = len(Data)
C = 2
'''

SBM  = StochasticBlockModel(U, C)
ELBO, converged, Post, LP = VB().run(SBM, Data)
print LP['resp']
