# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 13:31:31 2016

@author: JohnLi

edit: Huiting hong
2017-0302
"""

import numpy as np

        
def _log_multivariate_normal_density_diag(X, means, covars):
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr

def log_normal_prob(feats, means, covars):
    """
    feats(N*D)
    prior(K*1)
    means(K*D)
    covars(K*D)
    """
    N = 2
    K, D = means.shape

#    lpr = [- 0.5*sum(np.log(covars[:,k]))
#           - 0.5*np.sum(((feats-means[k,:])/covars[k,:])**2,1) 
#           for k in range(K)]
   
    lpr = []
    for k in range(K):
        delta = (feats-means[k,:])/(covars[k,:]**0.5)
        lpr.append(- 0.5*sum(np.log(covars[k,:])) - 0.5*np.sum(delta**2,1))
    return np.array(lpr).T


def predict_prob(feats, priors, means, covars):     
    logprob = log_normal_prob(feats, means, covars) + np.log(priors)
    lpr = np.exp(logprob.T - logprob.max(axis=1))
    responsibilities = lpr/np.sum(lpr, 0)
    return responsibilities.T
    
        
def FisherEncode(feats, means, covars, priors, square_root=False, normalized=False, improved=False, improved_v1=False, improved_v2=False, fast=False) :
    """
    Encode features given GMM parameters using Fisher encoding
    
    *** Output ***
    return: 1D encoded feature vector encodedFeats
    
    *** Input ***
    feat : n_samples, n_dim
    means : K_cluster, n_dim (K*D)
    covars : K_cluster, n_dim
    priors : K_cluster array

    *** Improve Setting ***
    improved : do once normalize along row-dir(among the FV of one data), using ±√xi / √(∑ xi^2)
    improved_v1 : do twice normalize along row-dir(among the FV of one data), using ±√xi / √(∑ xi^2)
    improved_v2 : do once normalize along row-dir(among the FV of one data), using xi-u/sigma
    
    """ 
#    try :
    N, D = feats.shape
    ##get posterior probabilities q for each data point and each component
    posteriors = predict_prob(feats, priors, means, covars)
    
    if fast:
        print(posteriors.shape)
        ind_most_likeli = np.argmax(posteriors, axis=1)
        posteriors = np.zeros(posteriors.shape)
        posteriors[:, ind_most_likeli] = 1
    
    us = np.empty(0)
    vs = np.empty(0)
    ## this one uses the formulation given by vlfeat
    for k, covars_k in enumerate(covars) :     
        means_k_rep = means[k, :].reshape((1, D)).repeat(N, axis=0)
        covars_k_rep = covars_k.reshape((1, D)).repeat(N, axis=0)
        post_k_rep = posteriors[:, k].reshape((N, 1)).repeat(D, axis=-1)
        
        delta = (feats-means_k_rep)/np.sqrt(covars_k_rep)
        
        uk = np.sum(post_k_rep*delta, axis=0)
        uk /= (N*np.sqrt(priors[k]))
        us = np.concatenate((us, uk))
        
        vk = np.sum(post_k_rep*(delta**2-1), axis=0)
        vk /= (N*np.sqrt(2*priors[k]))
        vs = np.concatenate((vs, vk))
        
    encodedFeats = np.concatenate((us, vs))
    
    if square_root:
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))
    
    if normalized:
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)
        
    if improved:
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))  
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)

    if improved_v1:
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))  
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))  
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)

    if improved_v2:
        mu = np.mean(encodedFeats)
        std = np.std(encodedFeats)
        encodedFeats = (encodedFeats-mu)/std
    
    return encodedFeats


if __name__=='__main__':
    ### some testing on fisher vector
    # from cyvlfeat.fisher import fisher
    N = 21 #data-num
    K = 3 #cluster-num
    D = 5 #feature-dim
    
    ### mean
    mu = np.zeros(D * K, dtype=np.float32)
    for i in range(D * K):
        mu[i] = i
    mu = mu.reshape(K, D).T
    
    ### covariance
    sigma2 = np.ones((D, K), dtype=np.float32)
    
    ### prior
    prior = (1.0 / K) * np.ones(K, dtype=np.float32)
    
    ### data
    x = np.zeros(D * N, dtype=np.float32)
    for i in range(D * N):
        x[i] = i
    x0 = x.reshape(D, N)
    x = x.reshape(N, D).T

    ### generate fisher vector    
    observed_t0 = FisherEncode(x0.T, mu.T, sigma2.T, prior, improved = False, fast=False) #dim = K*D
    observed_t = FisherEncode(x.T, mu.T, sigma2.T, prior, improved = False, fast=False)
    
    # observed_vl0 = fisher(x0, mu, sigma2, prior, False, improved = False,fast= False,verbose=True)
    # observed_vl = fisher(x, mu, sigma2, prior, False, improved = False,fast =False,verbose=True)
