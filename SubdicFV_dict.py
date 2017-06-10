# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 16:01:39 2017

@author: Huiting hong
"""
from __future__ import print_function
import os
import numpy as np
import csv
import glob
import time
import scipy.io as sio
import pandas as pd 
import pickle
from sklearn import mixture
from sklearn.svm import SVC
from sklearn import feature_selection
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from fisher_encode import FisherEncode
 
def Norm_Col(dic):
    # initial empty 2-D array preparing to store all 'values'
    rowN = len(dic)
    colN = dic.itervalues().next().shape[0]
    val_map = np.zeros(rowN*colN)
    val_map = val_map.reshape(rowN,colN)
    fname_lst = []

    for idx, (key,value) in enumerate(dic.items()):
        val_map[idx] = value
        fname_lst.append(key)

    # Do transpose in order to normalize dic. in columnwise
    val_map = val_map.T
    for i,ele in enumerate(val_map):
        ele = np.sign(ele)*np.sqrt(abs(ele)) 
        val_map[i] = ele/np.linalg.norm(ele)

    # In case of NaN
    val_map[np.isnan(val_map)] = 0

    # transpose back and save to dictionary
    val_map = val_map.T

    New_dic = {}
    for i,ele in enumerate(fname_lst):
        New_dic[ele] = val_map[i]
    return New_dic 

def main():
	## Set path
	main_path = '../'
	gmm_path = '../GMM_model'

	## Read features
	filepath_tra = main_path+"lld_train_final_large.pickle"
	labelpath_tra = main_path+"label_train_final_large.pickle"

	with open(filepath_tra, 'rb') as handle:
		Data_Train = pickle.load(handle)
	with open(labelpath_tra, 'rb') as handle:
		Label_Train = pickle.load(handle)

	## Decide which GMM, mixtureNum u want to use
	klist = [4,8,16,32,64,128,256]
	# gs = ['gen','labelC','labelNC']
	gs = ['gen','labelC','labelNC','unspvAA','unspvBB']
	for GMM_genarate_source in gs: # Choice: gen, labelC, labeNC, unspvAA, unspvBB
		for k in klist:

	# k = 64
	# GMM_genarate_source = 'labelNC'

			means = joblib.load(gmm_path +'/'+ GMM_genarate_source +'_'+ str(k) +'_mean.pkl') # K*1 
			covars = joblib.load(gmm_path+'/'+ GMM_genarate_source +'_'+ str(k) +'_cov.pkl') # D*K
			priors = joblib.load(gmm_path+'/'+ GMM_genarate_source +'_'+ str(k) +'_priors.pkl') # D*K  

			## FV of sub dictionary
			## FV for train
			save_dir = main_path+'FV/'+GMM_genarate_source+'_GMM'
			save_data = 'train'
			fv_stack = []
			dic_TrainFV = {}
			for fname, feature in Data_Train.iteritems():
				if Label_Train.get(fname) != None:
					Fv_feature = FisherEncode(feature, means.T, covars.T, priors, improved=True)
					dic_TrainFV[fname] = Fv_feature
				
			# column-wise normalization
			# dic_TrainFV = Norm_Col(dic_TrainFV)

			if not os.path.exists(save_dir):
			    os.makedirs(save_dir)
			joblib.dump(dic_TrainFV, save_dir+'/'+str(k)+'_'+save_data+'Fisher.pkl')
			print('finish --- ',save_data)


			## FV for devel
			testfilelist = glob.glob(main_path+'test/lld_test*.pickle')
			for i,filepath in enumerate(testfilelist):
				with open(filepath, 'rb') as handle:
					Data_Test = pickle.load(handle)
				save_data = os.path.basename(filepath)[:-7] #'lld_test*'
				fv_stack = []
				dic_TestFV = {}
				for fname, feature in Data_Test.iteritems():
					Fv_feature = FisherEncode(feature, means.T, covars.T, priors, improved=True)
					dic_TestFV[fname] = Fv_feature

				# column-wise normalization
				# dic_TestFV = Norm_Col(dic_TestFV)

				joblib.dump(dic_TestFV, save_dir+'/'+str(k)+'_'+save_data+'_Fisher.pkl')
				print('finish --- ',save_data)

if __name__ == '__main__':
	main()
