# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 11:24:03 2017

@author: Huiting hong
"""
from __future__ import print_function
import os
import numpy as np
import csv
import glob
import time
import random
import scipy.io as sio
import pandas as pd 
import pickles
from sklearn import mixture
from sklearn.svm import SVC
from sklearn import feature_selection
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from fisher_encode import FisherEncode
from Encode import BagOfWords, BOW, fID, Partial_read


def train_GMM(k, Samples):
    g = GaussianMixture(n_components=k, covariance_type='diag')
    g.fit(Samples)
    gmm_mean = np.array((g.means_).T, dtype='float32')
    gmm_cov = np.array((g.covariances_ ).T, dtype='float32')
    gmm_priors = np.array(g.weights_, dtype='float32')
    return gmm_mean, gmm_cov, gmm_priors

def save_GMM(GMM_para, main_dir, save_name, k):
    
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    joblib.dump(GMM_para[0], main_dir +'/'+ save_name +'_'+ str(k) +'_mean.pkl')
    joblib.dump(GMM_para[1], main_dir +'/'+ save_name +'_'+ str(k) +'_cov.pkl')
    joblib.dump(GMM_para[2], main_dir +'/'+ save_name +'_'+ str(k) +'_priors.pkl')
    # print(main_dir +'/'+ save_name +'_'+ str(k) +'_gmm')

def ConsVal(dic):
	DIM = dic.itervalues().next().shape[1]
	D_mat = np.empty([0, DIM])
	for index, (key, value) in enumerate(sorted(dic.items())):
	    D_mat = np.concatenate([D_mat, value], axis=0)
	return D_mat

def main():
	## Set path
	main_path = '../'
	gmm_path = './GMM_model'

	## Read features
	filepath_tra = main_path+"lld_train_final_large.pickle"
	labelpath_tra = main_path+"label_train_final_large.pickle"

	# filepath_tra = main_path+"TEST.pickle"
	# labelpath_tra = main_path+"Label_TEST.pickle"


	# Open train data & train label
	with open(filepath_tra, 'rb') as handle:
		Data_Train = pickle.load(handle)
	with open(labelpath_tra, 'rb') as handle:
		Label_Train = pickle.load(handle)

	# All data vals
	txt_train = ConsVal(Data_Train)

	## Separate C,NC data vals
	C_dict = {}
	NC_dict = {}
	for fname, CNC in Label_Train.iteritems():
		if CNC == 1:
			C_dict[fname] = Data_Train.get(fname)
		else :
			NC_dict[fname] = Data_Train.get(fname)
	txt_trainC = ConsVal(C_dict)
	txt_trainNC = ConsVal(NC_dict)

	## Set mixture Num
	GMM_Mixures = [4, 8, 16, 32, 64, 128, 256]

	for k in GMM_Mixures: 
		print('k = ',k)
		## GMM
		main_dir = main_path + '/GMM_model'

		# (General generated)
		## generate GMM model of train_all 
		GMM_para = train_GMM(k, txt_train)
		save_name = 'gen'
		save_GMM(GMM_para, main_dir, save_name, k)
		# means = GMM_para[0]
		# covars = GMM_para[1]
		# priors = GMM_para[2]
		print('general-model finish')
		

		# (Label generated)
		## generate GMM model of train_C part
		GMM_para = train_GMM(k, txt_trainC)
		save_name = 'labelC'
		save_GMM(GMM_para, main_dir, save_name, k)
		print('labelC-model finish')

		## generate GMM model of train_NC part
		GMM_para = train_GMM(k, txt_trainNC)
		save_name = 'labelNC'
		save_GMM(GMM_para, main_dir, save_name, k)
		print('labelNC-model finish')


		# (Unsupervised generated)
		## use kmeans to generate clusterAA and clusteBB
		## Create Samples prepare to initial kmeans center points
		Samples = txt_train

		k_cluster = 2
		B = BagOfWords(k_cluster)  
		BOW.voc = B.train(Samples)
		A_cluster = []
		B_cluster = []
		for fname, features in Data_Train.iteritems():
			BOW.words, BOW.hist = B.makeHistogram(features)
			if BOW.hist[0] > BOW.hist[1]: # vote as cluster A
				A_cluster.append(fname)
			elif BOW.hist[0] < BOW.hist[1]: # vote as cluster A
				B_cluster.append(fname)
			else: # randomly vote for A or B
				tar = random.sample([0,1],1)
				if tar == 0:
					A_cluster.append(fname)
				else:
					B_cluster.append(fname)


		## base on the A or B cluster result to get their features
		txt_unsuperviseAA = []
		txt_unsuperviseBB = []
		for i,ele in enumerate(A_cluster):
			fea = Data_Train.get(ele)
			if txt_unsuperviseAA == []:
				txt_unsuperviseAA = fea
			else:
				txt_unsuperviseAA = np.concatenate((txt_unsuperviseAA,fea),axis = 0)

		for i,ele in enumerate(B_cluster):
			fea = Data_Train.get(ele)
			if txt_unsuperviseBB == []:
				txt_unsuperviseBB = fea
			else:
				txt_unsuperviseBB = np.concatenate((txt_unsuperviseBB,fea),axis = 0)

		# print(txt_unsuperviseAA)
		# print(txt_unsuperviseBB)

		## generate GMM model of train_AA part
		GMM_para = train_GMM(k, txt_unsuperviseAA)
		save_name = 'unspvAA'
		save_GMM(GMM_para, main_dir, save_name, k)
		print('unsupervise-cluster1-model finish')

		## generate GMM model of train_BB part
		GMM_para = train_GMM(k, txt_unsuperviseBB)
		save_name = 'unspvBB'
		save_GMM(GMM_para, main_dir, save_name, k)
		print('unsupervise-cluster2-model finish')
	

if __name__ == '__main__':
	main()


