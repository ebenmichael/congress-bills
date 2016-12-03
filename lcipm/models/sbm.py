"""
Implementation of a Bayesian Stochastic Block Model
Author: Jake Soloff
"""

import numpy as np
from math import log
from scipy.special import digamma
from scipy.special import gammaln as loggamma
from abstractmodel import AbstractModel
from modelnodes import AbstractNode
from ValBag import ValBag
from copy import deepcopy as copy

import json

class StochasticBlockModel(AbstractModel):

    def __init__(self, dim, lam_prior=None, gam_prior=None):
        """Constructor"""
        self.dim = dim
        self.resp_node = RespNode(dim)
        self.rate_node = RateNode(dim, lam_prior)
        self.prop_node = PropNode(dim, gam_prior)
        
        # point the nodes to the other nodes
        self.resp_node.assign_params(self.rate_node, self.prop_node)
        self.rate_node.assign_params(self.resp_node)
        self.prop_node.assign_params(self.resp_node)

        self.nodes = {"resp": self.resp_node,
                      "coexp_rates": self.rate_node,
                      "proportions": self.prop_node}
        self.local_parameters = ["resp"]
        self.global_parameters = ["coexp_rates", "proportions"]

    def assign_data(self, data):
        """Giving data to a model"""
        data = np.array(data, dtype=int)
        self.data = data
        for node in self.local_parameters:
            self.nodes[node].assign_data(data)

    def init_v_params(self):
        """Initialize variational parameters"""
        pass

    def calc_SS(self):
        """Compute the sufficient statistics for the global parameters"""
        Data = self.data
        resp = self.resp_node.resp
        SS   = {}

        Nk  = np.sum(resp, axis=0)
        SS['N_k_full'] = Nk

        Nkl = np.outer(Nk,Nk) 
        for k in range(C):
            for l in range(C):
                Nkl[k,l] -= np.dot(resp[:,k],resp[:,l]) 
        SS['N_kl_full'] = Nkl

        Skl = np.zeros((C,C))
        for k in xrange(C):
            for l in xrange(C):
                for u in xrange(U):
                    for v in xrange(U):
                        if u != v:
                            Skl[k,l] += resp[u,k]*resp[v,l]*Data[u,v]
        SS['S_kl_full'] = Skl
        self.SS = SS

        return SS

    def calc_data_elbo(self):
        """Compute ELBO conrtibution from likelihood"""
        elbo = 0.
        lam_prior = self.rate_node.prior
        lam_post  = self.rate_node.post

        """
        elbo += np.sum(lam_prior[0]*np.log(lam_prior[1]) 
                    - lam_post[0,:,:]*np.log(lam_post[1,:,:]) 
                    - loggamma(lam_prior[0]) + loggamma(lam_post[0,:,:]))
        elbo += np.sum(self.SS['S_kl_full'] + lam_prior[0] - lam_post[0,:,:]* 
                    (digamma(lam_post[0,:,:]) - np.log(lam_post[1,:,:])))
        elbo -= np.sum(self.SS['N_kl_full'] + lam_prior[1] - lam_post[1,:,:]* 
                    (lam_post[0,:,:] / lam_post[1,:,:]))
        """
        _, C = self.dim
        elbo  = np.zeros((C,C))
        for k in xrange(C):
            for l in xrange(C):
                elbo[k,l] += lam_prior[0]*log(lam_prior[1]) - lam_post[0,k,l]*log(lam_post[1,k,l]) \
                                                 - loggamma(lam_prior[0]) + loggamma(lam_post[0,k,l])
                elbo[k,l] += (self.SS['S_kl_full'][k,l] + lam_prior[0] - lam_post[0,k,l])* \
                                                      (digamma(lam_post[0,k,l]) - log(lam_post[1,k,l]))
                elbo[k,l] -= (self.SS['N_kl_full'][k,l] + lam_prior[1] - lam_post[1,k,l])* \
                                                               (lam_post[0,k,l] / log(lam_post[1,k,l]))
        return np.sum(elbo)

    ## evaluate the evidence lower bound -- need to transfer this to nodes
    def calcELBO(self, Data, SS, LP):
        C = self.C
        elbo  = np.zeros((C,C))
        Post  = self.Post
        Prior = self.Prior
        
        elbo += Prior.lambd[0]*log(Prior.lambd[1]) - Post.lam0*log(Post.lam1) \
                                         - loggamma(Prior.lambd[0]) + loggamma(Post.lam0)
        elbo += (SS.S_kl_full[k,l] + Prior.lambd[0] - Post.lam0[k,l])* \
                                              (digamma(Post.lam0[k,l]) - log(Post.lam1[k,l]))
        elbo -= (SS.N_kl_full[k,l] + Prior.lambd[1] - Post.lam1[k,l])* \
                                                       (Post.lam0[k,l] / log(Post.lam1[k,l]))
        ElogPi = E_log_pi(Post.gamma) 

        # contributions to the ELBO in order: L_data, L_ent, L_local, L_global
        return np.sum(elbo) \
                + SS.H \
                + np.dot(ElogPi,SS.N_k_full) \
                + loggamma(Prior.gamma*C) - C*loggamma(Prior.gamma) \
                    + np.sum(loggamma(Post.gamma)) - loggamma(np.sum(Post.gamma)) - np.dot(ElogPi,Prior.gamma-Post.gamma)

class RespNode(AbstractNode):

    def __init__(self, dim):
        self.dim = dim

    def assign_data(self, data):
        """Give the node whatever data it needs"""
        self.data = data

    def assign_params(self, rate_node, prop_node):
        self.rate_node = rate_node
        self.prop_node = prop_node

    def get_v_params(self):
        """Return the variational parameters"""
        pass

    def init_v_params(self):
        """Initialize the variational parameters"""
        U, C = self.dim
        #resp = np.random.random((U,C))
        #for u in range(U):
        #    resp[u,:] /= np.sum(resp[u,:])

        with open('../../data/combined_data/pos_to_party.json') as f:
            dct = json.load(f)

        U = len(dct)
        resp_ = np.zeros((U,2))
        for u,p in dct.items():
            resp_[u,int(p=='R')] = 1.0
        print resp_

        self.resp = resp_

    def vi_update(self):
        """Update variational parameters"""
        if not(hasattr(self,'resp')):
            self.init_v_params()
            return
        resp = self.resp
        resp_= copy(resp)
        NN = 0
        while True:
            NN += 1
            resp__= copy(resp_)
            resp_ = copy(resp)
            resp  = self.updateResp()
            if NN > 2: # np.allclose(resp__, resp) or 
                break
        self.resp = resp

    def updateResp(self):
        U,C  = self.dim
        Data = self.data
        resp = self.resp

        resp_   = np.zeros((U,C))
        logresp_= np.zeros((U,C))

        ElogP  = digamma(self.rate_node.post[0,:,:]) - np.log(self.rate_node.post[1,:,:])
        EP     = self.rate_node.post[0,:,:] / self.rate_node.post[1,:,:]
        ElogPi = digamma(self.prop_node.post) - digamma(np.sum(self.prop_node.post))

        for u in range(U):
            for k in range(C):
                for l in range(C):
                    for v in range(U):
                        if v != u:
                            logresp_[u,k] += resp[v,l]*(Data[u,v]*ElogP[k,l] - EP[k,l])
            logresp_[u,:] += ElogPi
        for u in range(U):
            resp_[u,:]     = np.exp(logresp_[u,:] - np.max(logresp_[u,:]))
            resp_[u,:]    /= np.sum(resp_[u,:])
        return resp_

    def calc_elbo(self):
        """Compute the contribution to elbo from prior/entropy"""
        resp = self.resp
        H = -resp*np.log(resp)
        H = np.sum(H[H < float('Inf')])

        Nk  = np.sum(resp, axis=0)
        ElogPi = digamma(self.prop_node.post) - digamma(np.sum(self.prop_node.post))

        return H + np.dot(ElogPi,Nk)

class RateNode(AbstractNode):

    def __init__(self, dim, lam_prior):
        self.dim   = dim
        self.prior = lam_prior

    def assign_data(self):
        """Give the node whatever data it needs"""
        raise NotImplementedError

    def assign_params(self, resp_node):
        self.resp_node = resp_node

    def get_v_params(self):
        """Return the variational parameters"""
        pass

    def init_v_params(self):
        """Initialize the variational parameters"""
        raise NotImplementedError

    def vi_update(self, SS):
        """Update variational parameters"""
        C = self.dim[1]
        self.post = np.zeros((2,C,C))
        self.post[0,:,:] = self.prior[0] + SS['S_kl_full']
        self.post[1,:,:] = self.prior[1] + SS['N_kl_full']

    def calc_elbo(self):
        """Compute the contribution to elbo from prior/entropy"""
        return 0.

class PropNode(AbstractNode):

    def __init__(self, dim, gam_prior):
        self.dim   = dim
        self.prior = gam_prior

    def assign_data(self):
        """Give the node whatever data it needs"""
        raise NotImplementedError

    def assign_params(self, resp_node):
        self.resp_node = resp_node

    def get_v_params(self):
        """Return the variational parameters"""
        pass

    def init_v_params(self):
        """Initialize the variational parameters"""
        raise NotImplementedError

    def vi_update(self, SS):
        """Update variational parameters"""
        self.post = self.prior + SS['N_k_full']

    def calc_elbo(self):
        """Compute the contribution to elbo from prior/entropy"""
        _, C = self.dim
        ElogPi = digamma(self.post) - digamma(np.sum(self.post))
        elbo = 0.
        elbo += loggamma(self.prior*C) - C*loggamma(self.prior) \
                    + np.sum(loggamma(self.post)) - loggamma(np.sum(self.post)) \
                    - np.dot(ElogPi,self.prior-self.post)
        return elbo

from VB import VB

import csv
fn = '../../data/combined_data/membership.dat'
Data = np.loadtxt(fn)[:,:]
Data = np.dot(Data,Data.T)

U = len(Data)
C = 2

"""
U = 100
C = 2
Data = np.zeros((U,U))
pi   = np.array([.5,.5])

M    = np.random.choice(C,U,p=pi)
P    = 2.*np.random.random((C,C)) + 2*np.eye(C) 
for u in xrange(U):
  for v in xrange(u):
      Data[u,v] = np.random.poisson(P[M[u],M[v]])
      Data[v,u] = Data[u,v]

M_ = np.zeros((U,C))
for u in range(len(M)):
   M_[u,M[u]] = 1.
#print M_
"""

SBM  = StochasticBlockModel((U, C), (1.,1.),1.)
ELBO, converged = VB().run(SBM, Data)

#np.savetxt('eli.dat', SBM.resp_node.resp)

#print np.loadtxt('eli.dat').sum(0)

print np.round(SBM.resp_node.resp,3)
#print np.sum(SBM.resp_node.resp - M_,0)