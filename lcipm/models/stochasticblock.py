"""
Implementation of a Bayesian Stochastic Block Model
Author: Jake Soloff
"""

import numpy as np
from math import log
from scipy.special import digamma
from scipy.special import gammaln as loggamma
from abstractmodel import AbstractModel
from ValBag import ValBag
from copy import deepcopy as copy

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
			gamma = np.array([.1]) 
		if lambd is None:
			lambd = np.array([.01,.01])
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

		resp = LP['resp']
		resp_= copy(resp)
		NN = 0
		while True:
			NN += 1
			resp__= copy(resp_)
			resp_ = copy(resp)
			resp  = self.updateResp(Data,resp)
			if np.allclose(resp__, resp) or NN > 20: 
				break
			#if NN > 20:
			#	print NN, resp
			#print resp

		LP['resp'] = resp
		return LP

	def updateResp(self,Data,resp):
		C = self.C
		U = self.U
		Post  = self.Post
		Prior = self.Prior

		resp_   = np.zeros((U,C))
		logresp_= np.zeros((U,C))

		ElogP = np.zeros((C,C))
		EP    = np.zeros((C,C)) # post.lam0 / post.lam1
		for k in range(C):
			for l in range(C):
				ElogP[k,l] = (digamma(Post.lam0[k,l]) - log(Post.lam1[k,l])) 
				EP[k,l] = (Post.lam0[k,l] / Post.lam1[k,l])

		ElogPi = E_log_pi(Post.gamma)
		for u in range(U):
			for k in range(C):
				for l in range(C):
					for v in range(U):
						if v != u:
							logresp_[u,k] += resp[v,l]*(Data[u,v]*ElogP[k,l] - EP[k,l])
			logresp_[u,:] += ElogPi
			resp_[u,:]     = np.exp(logresp_[u,:])
			resp_[u,:]    /= np.sum(resp_[u,:])
		return resp_

	## initialize LP dictionary
	## completely random init
	def initLP(self, Data):
		LP = dict()
		resp = np.random.random((U,C))
		for u in range(self.U):
			resp[u,:] /= np.sum(resp[u,:])
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
		for k in range(C):
			for l in range(C):
				Nkl[k,l] -= np.dot(resp[:,k],resp[:,l]) 
		SS.setField('N_kl_full', Nkl)

		Skl = np.zeros((C,C))
		for k in xrange(C):
			for l in xrange(C):
				for u in xrange(U):
					for v in xrange(U):
						if u != v:
							Skl[k,l] += resp[u,k]*resp[v,l]*Data[u,v]
				#Skl[k,l] = np.sum(np.outer(resp[:,k],resp[:,l]) * Data) - np.trace(np.outer(resp[:,k],resp[:,l]) * Data)
		SS.setField('S_kl_full', Skl)
		H = -resp*np.log(resp)
		SS.setField('H',np.sum(H[H < float('Inf')]))

		# print 'Nk', Nk
		# print 'Nkl',Nkl
		# print 'Skl',Skl
		# print 'ent',H
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
											     - loggamma(Prior.lambd[0]) + loggamma(Post.lam0[k,l])
				elbo[k,l] += (SS.S_kl_full[k,l] + Prior.lambd[0] - Post.lam0[k,l])* \
													  (digamma(Post.lam0[k,l]) - log(Post.lam1[k,l]))
				elbo[k,l] -= (SS.N_kl_full[k,l] + Prior.lambd[1] - Post.lam1[k,l])* \
															   (Post.lam0[k,l] / log(Post.lam1[k,l]))
		ElogPi = E_log_pi(Post.gamma) 

		# contributions to the ELBO in order: L_data, L_ent, L_local, L_global
		return np.sum(elbo) \
				+ SS.H \
				+ np.dot(ElogPi,SS.N_k_full) \
				+ loggamma(Prior.gamma*C) - C*loggamma(Prior.gamma) \
					+ np.sum(loggamma(Post.gamma)) - loggamma(np.sum(Post.gamma)) - np.dot(ElogPi,Prior.gamma-Post.gamma)

def E_log_pi(gam):
	return digamma(gam) - digamma(np.sum(gam))

from VB import VB

# generate toy dataset
# U, C = 4, 2
# Data  = np.array([[10., 8., 1., 1.],
# 				  [8., 10., 1., 1.],
# 				  [1., 1.,  10.,9.],
# 				  [1., 1.,  9.,10.]])
# U = 100
# C = 2
# Data = np.zeros((U,U))
# pi   = np.array([.5,.5])#np.ones((1,C))/C#np.array([.5,.5,.5])#np.random.dirichlet([1.]*C)

# M    = np.random.choice(C,U,p=pi)
# P    = 2.*np.random.random((C,C)) + 2*np.eye(C)    #np.array([[7.,.2,.5],[.2,7.,1.], [.5,1.,7.]]) 
# for u in xrange(U):
# 	for v in xrange(u):
# 		Data[u,v] = np.random.poisson(P[M[u],M[v]])
# 		Data[v,u] = Data[u,v]


import csv
fn = '../data/caucus/membership_110.csv'
fn = '../data/combined_data/membership.dat'
# l = []
# with open(fn,'r') as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		l.append([int(s) for s in row[1:]])
# Data = np.array(l[1:])
# Data = np.dot(Data,Data.T)
Data = np.loadtxt(fn)[:,:100]
Data = np.dot(Data,Data.T)

U = len(Data)
C = 2



SBM  = StochasticBlockModel(U, C)
ELBO, converged, Post, LP = VB().run(SBM, Data)
print np.round(LP['resp'],3)
#print LP['resp']
print np.sum(LP['resp'],0)

#M_ = np.zeros((U,C))
#for u in range(len(M)):
#	M_[u,M[u]] = 1.
#print M_


#resp = LP['resp']
#print (M_ - resp).sum(0)
#print np.sum(M_,axis=0), np.sum(LP['resp'],axis=0)
