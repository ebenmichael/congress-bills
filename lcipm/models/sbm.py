"""
Implementation of a Bayesian Stochastic Block Model
Author: Jake Soloff, edited by Eli Ben-Michael
"""

import numpy as np
from math import log
from scipy.special import digamma
from scipy.special import gammaln as loggamma
from .abstractmodel import AbstractModel
from .modelnodes import AbstractNode
from copy import deepcopy as copy
from six.moves import xrange
import os


class StochasticBlockModel(AbstractModel):

    def __init__(self, c, lam_prior=(1.0, 1.0), gam_prior=1.0):
        """Constructor"""
        self.C = c
        self.resp_node = RespNode(c)
        self.rate_node = RateNode(c, lam_prior)
        self.prop_node = PropNode(c, gam_prior)

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
        self.U = data.shape[0]

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

        C = self.C
        U = self.U
        Nkl = np.outer(Nk,Nk) 
        for k in range(self.C):
            for l in range(self.C):
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
        C = self.C
        elbo  = np.zeros((C,C))
        for k in xrange(C):
            for l in xrange(C):
                elbo[k,l] += lam_prior[0]*log(lam_prior[1]) \
                             - lam_post[0,k,l]*log(lam_post[1,k,l]) \
                             - loggamma(lam_prior[0])  \
                             + loggamma(lam_post[0,k,l])
                elbo[k,l] += (self.SS['S_kl_full'][k,l] + lam_prior[0] \
                              - lam_post[0,k,l])* \
                              (digamma(lam_post[0,k,l]) - log(lam_post[1,k,l]))
                elbo[k,l] -= (self.SS['N_kl_full'][k,l] + lam_prior[1] \
                              - lam_post[1,k,l])* \
                              (lam_post[0,k,l] / log(lam_post[1,k,l]))
        return np.sum(elbo)

    def save(self, outdir):
        """Save model parameters
        Args:
            outdir: string, outdir to write to
        """
        # responsibilities
        np.savetxt(os.path.join(outdir, "responsibility.dat"),
                   self.resp_node.resp)
        # proportions
        np.savetxt(os.path.join(outdir, "proportion.dat"),
                   self.prop_node.post)
        # rates
        np.savetxt(os.path.join(outdir, "coexp_rate.dat"),
                   self.rate_node.post.flatten())


class RespNode(AbstractNode):

    def __init__(self, n_communities):
        self.n_communities = n_communities

        # assume that the model type is StochasticBlockModel
        self.model_type = "StochasticBlockModel"

    def assign_data(self, data):
        """Give the node whatever data it needs"""
        self.data = data
        self.n_users = data.shape[0]

    def assign_params(self, rate_node, prop_node):
        self.rate_node = rate_node
        self.prop_node = prop_node

    def assign_lcipm_params(self, cmean_node, ip_node):
        """Point the RespNode to the ideal points and community means"""
        self.cmean_node = cmean_node
        self.ip_node = ip_node

    def set_model(self, model_type):
        """Tell RespNode which model it's in so it does the right updates"""
        self.model_type = model_type

    def get_v_params(self):
        """Return the variational parameters"""
        pass

    def init_v_params(self):
        """Initialize the variational parameters"""
        U, C = self.n_users, self.n_communities
        resp = np.random.random((U,C))
        for u in range(U):
            resp[u,:] /= np.sum(resp[u,:])
        self.resp = resp

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
            resp  = self.updateResp()
            if NN > 2: # np.allclose(resp__, resp) or 
                break
        self.resp = resp

    def updateResp(self):
        U,C  = self.n_users, self.n_communities
        Data = self.data
        resp = self.resp

        resp_   = np.zeros((U,C))
        logresp_= np.zeros((U,C))

        ElogP  = digamma(self.rate_node.post[0,:,:])\
                 - np.log(self.rate_node.post[1,:,:])
        EP     = self.rate_node.post[0,:,:] / self.rate_node.post[1,:,:]
        ElogPi = digamma(self.prop_node.post) - digamma(np.sum(self.prop_node.post))

        for u in range(U):
            for k in range(C):
                for l in range(C):
                    for v in range(U):
                        if v != u:
                            logresp_[u,k] += resp[v,l] * (Data[u,v] * ElogP[k,l] \
                                                          - EP[k,l])
                if self.model_type == "LCIPM":
                    # check that the user is in the interaction data
                    # if it is then do the lcipm update
                    if u in self.ip_node.data:
                        varx = self.ip_node.prior_var
                        ip_dim = self.ip_node.dim
                        ip_v_var = self.ip_node.v_var
                        ip_v_mean = self.ip_node.v_mean[u, :]
                        cmean_v_var = self.cmean_node.v_var
                        cmean_v_mean = self.cmean_node.v_mean[k, :]
                        p = -1 / (2 * varx)
                        s = ip_dim * (ip_v_var + cmean_v_var)
                        s -= np.sum((ip_v_mean - cmean_v_mean) ** 2)
                        s2 = -np.log(2 * np.pi * varx) * ip_dim / 2
                        logresp_[u, k] += s2 + p * s
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

    def __init__(self, n_communities, lam_prior):
        self.prior = lam_prior
        self.n_communities = n_communities

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
        C = self.n_communities
        self.post = np.zeros((2,C,C))
        self.post[0,:,:] = self.prior[0] + SS['S_kl_full']
        self.post[1,:,:] = self.prior[1] + SS['N_kl_full']

    def calc_elbo(self):
        """Compute the contribution to elbo from prior/entropy"""
        return 0.


class PropNode(AbstractNode):

    def __init__(self, n_communities, gam_prior):
        self.prior = gam_prior
        self.n_communities = n_communities

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
        C = self.n_communities
        ElogPi = digamma(self.post) - digamma(np.sum(self.post))
        elbo = 0.
        elbo += loggamma(self.prior*C) - C*loggamma(self.prior) \
                    + np.sum(loggamma(self.post)) - loggamma(np.sum(self.post)) \
                    - np.dot(ElogPi,self.prior-self.post)
        return elbo
