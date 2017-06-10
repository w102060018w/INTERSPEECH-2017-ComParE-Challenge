# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:09:13 2017

@author: Selly
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import time as T
#import tool

CLUSTERS = 64			# cluster number of kmeans (ex: 64, 128, 256
MIXTURE = 32			# mixture number of gaussian
PERCENTILE = 20			# Percent of features to keep. (ex: 20~100%
C = 1					# penalty parameter of SVM (ex: 0.1, 1, 10
KERNEL = 'linear'		# SVM kernel
#B = tool.BagOfWords(CLUSTERS)
SELECTOR = SelectPercentile(f_classif, PERCENTILE)

# #==============================================================================
# #   Training Data Phase
# #       collect kmeans model with calculated centroid from train.csv features
# #==============================================================================
# # Train KMeans Model
# TYPE = 'train'
# raw = tool.readRaw(TYPE)
# B.train(raw.drop('name', axis=1))
# with open('bow_k'+str(CLUSTERS)+'.pkl', 'wb') as f:
# 	pickle.dump(B.classifier, f)
# # Encode BOW histogram
# feature, label_table = tool.classFeature(raw, TYPE)
# hist_train = list()
# label_train = list()
# for key, value in feature.iteritems():
# 	_, hist = B.makeHistogram(np.array(value))
# 	hist_train.append(hist)
# 	label_train.append(label_table[key])
# with open('hist_train.pkl', 'wb') as f:
# 	pickle.dump(hist_train, f)
# with open('label_train.pkl', 'wb') as f:
# 	pickle.dump(label_train, f)
#
# selected_train = SELECTOR.fit_transform(hist_train, label_train)
# with open('sel_p'+str(PERCENTILE)+'.pkl', 'wb') as f:
# 	pickle.dump(SELECTOR, f)
#
# # Train SVM model
# SVM.fit(selected_train, label_train)
#
# #==============================================================================
# #   Testing Phase
# #==============================================================================
# feature, label_table = tool.readFile('devel')
# hist_test = list()
# label_test = list()
# B = tool.loadBOW(CLUSTERS)
# for key, value in feature.iteritems():
# 	_, hist = B.makeHistogram(np.array(value))
# 	hist_test.append(hist)
# 	label_test.append(label_table[key])
# with open('hist_devel.pkl', 'wb') as f:
# 	pickle.dump(hist_test, f)
# with open('label_devel.pkl', 'wb') as f:
# 	pickle.dump(label_test, f)
#
# selected_test = SELECTOR.transform(hist_test)
# # prediction
# result = SVM.predict(selected_test)
#
# accu = accuracy_score(label_test, result)*100
# uar = recall_score(label_test, result, average='macro')*100
# cm = confusion_matrix(label_test, result)
#
#

## ==============================================================================
##   encode with GMM to Fisher vector
## ==============================================================================
#MIXTURES = [16]#,32,64]#,128,256]#2, 4, 8, 16, 32, 64]
#fusion = True
#for MIXTURE in MIXTURES:
#	TYPE = 'train'
#	feature = tool.readMat(MIXTURE, TYPE, fusion=fusion)
#	label_train = tool.labelFeature(feature[0], TYPE)
#	fv_train = feature.drop(0, axis=1)
#	TYPE = 'devel'
#	feature = tool.readMat(MIXTURE, TYPE, all=False, fusion=fusion)
#	label_test = tool.labelFeature(feature[0], TYPE)
#	fv_test = feature.drop(0, axis=1)
#
#	f = open('./test/m0'+str(MIXTURE)+'_2.csv', 'w')
#	f.write(',0.1,,,1,,,10,,\n')
#	f.write('PERCENTILE,accu,uar,cm,accu,uar,cm,accu,uar,cm\n')
#
#	for PERCENTILE in np.arange(10, 110, 10):
#		SELECTOR = SelectPercentile(f_classif, PERCENTILE)
#		selected_train = SELECTOR.fit_transform(fv_train, label_train)
#		# with open('sel_GMM.pkl', 'wb') as f:
#		# 	pickle.dump(SELECTOR, f)
#		# Testing
#		selected_test = SELECTOR.transform(fv_test)
#
#		f.write(str(PERCENTILE))
#		for C in [0.1, 1, 10]:
#			SVM = SVC(C, KERNEL, class_weight='balanced')
#			# train SVM model
#			SVM.fit(selected_train, label_train)
#			# with open('svm_GMM.pkl', 'wb') as f:
#			# 	pickle.dump(SVM, f)
#			# prediction
#			result = SVM.predict(selected_test)
#
#
#			print PERCENTILE, C
#			accu = accuracy_score(label_test, result)*100
#			uar = recall_score(label_test, result, average='macro')*100
#			auc = roc_auc_score(label_test, result)*100
#			cm = confusion_matrix(label_test, result, labels=[True, False])
#			print ('acc: {:.4f}\t uar: {:.4f}\n cm:\n{}'.format(accu, uar, cm))
#			f.write(',{},{},{}'.format(accu, uar, str(cm).replace('\n', '')))
#		f.write('\n')
#	f.close()
#

# Read FV
def ConsVal(dic,label):
#	DIM = dic.itervalues().next().shape[0] # 131*4*2, which 4 is k_val
	D_mat = []
	Label_mat = []
	for index, (key, value) in enumerate(sorted(dic.items())):
		Label_mat.append(label.get(key))
		D_mat.append(value)
	D_mat = np.array(D_mat)
	Label_mat = np.array(Label_mat)
	return D_mat, Label_mat
#===============================================================================
#   different classifier
# ==============================================================================
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import BernoulliNB as BNB

# Set path
FV_pth = './FV/'
Label_pth = './'
trainFV_fn = 'trainFisher.pkl'
testFV_fn = 'testFisher.pkl'
trainLabel_fn = 'label_train_final_large.pickle'
testLabel_fn  = 'label_train_final_large.pickle'

# Load FV and data labels
with open(Label_pth+trainLabel_fn, 'rb') as handle:
	train_label = pickle.load(handle)
with open(Label_pth+testLabel_fn, 'rb') as handle:
	test_label = pickle.load(handle)


# FVS = ['gen', 'labelC', 'labelNC']
FVS = ['unspvAA', 'unspvBB']
MIXTURES = [8,16,32,64]#16
for MIXTURE in MIXTURES:
	model = list()
	for FV in FVS:
		path = FV_pth+FV+'_GMM/'+str(MIXTURE)+'_'
		# Get values in dict and merge all elements into an array.
		fv_train = joblib.load(path+trainFV_fn)
		fv_test  = joblib.load(path+testFV_fn)
		fv_train, label_train = ConsVal(fv_train,train_label)
		fv_test,  label_test  = ConsVal(fv_test,test_label)
   

		########## SVM
		tStart = T.time()
		M = 'SVM'
		print 'FV_{}\tmix={}\tSVM'.format(FV,MIXTURE)
		for i,C in enumerate([0.1,1,10,100]):
			model.append(FV+'_'+M+str(C))
			clf = SVC(C, KERNEL, probability=True)#, class_weight='balanced')
			clf.fit(fv_train, label_train)
			prob_train = clf.decision_function(fv_train)
			prob_test  = clf.decision_function(fv_test)
			if (i == 0):# and (FV=='gen'):
				score_train = [prob_train]
				score_test  = [prob_test]
			else:
				score_train = np.append(score_train, [prob_train], axis=0)
				score_test  = np.append(score_test , [prob_test] , axis=0)
		###
		score_train = np.transpose(score_train)
		score_test  = np.transpose(score_test)
		joblib.dump(score_train, FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_trainFisher.pkl')
		joblib.dump(score_test , FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_testFisher.pkl')
		###
		print 'execution time:{:.2f}s'.format(T.time()-tStart)
		########## Random Forest
		tStart = T.time()
		M = 'RF'
		print 'FV_{}\tmix={}\tRandom Forest'.format(FV,MIXTURE)
		for i,TREE in enumerate([10,50,100,200,500]):
			model.append(FV+'_'+M+str(TREE))
			clf = RF(TREE)#, class_weight='balanced')
			clf.fit(fv_train, label_train)
			prob_train = np.delete(clf.predict_proba(fv_train),1, axis=1).flatten()
			prob_test  = np.delete(clf.predict_proba(fv_test) ,1, axis=1).flatten()
			if (i == 0):# and (FV=='gen'):
				score_train = [prob_train]
				score_test  = [prob_test]
			else:
				score_train = np.append(score_train, [prob_train], axis=0)
				score_test  = np.append(score_test , [prob_test] , axis=0)
		###
		score_train = np.transpose(score_train)
		score_test  = np.transpose(score_test)
		joblib.dump(score_train, FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_trainFisher.pkl')
		joblib.dump(score_test , FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_testFisher.pkl')
		###
		print 'execution time:{:.2f}s'.format(T.time()-tStart)
   
		########## AdaBoost
		tStart = T.time()
		M = 'AB'
		print 'FV_{}\tmix={}\tAdaboost'.format(FV,MIXTURE)
		for i,TREE in enumerate([10,50,100,200,500]):
			model.append(FV+'_'+M+str(TREE))
			clf = AB(n_estimators=TREE)
			clf.fit(fv_train, label_train)
			prob_train = clf.decision_function(fv_train)
			prob_test  = clf.decision_function(fv_test)
			if (i == 0):# and (FV=='gen'):
				score_train = [prob_train]
				score_test  = [prob_test]
			else:
				score_train = np.append(score_train, [prob_train], axis=0)
				score_test  = np.append(score_test , [prob_test] , axis=0)
		###
		score_train = np.transpose(score_train)
		score_test  = np.transpose(score_test)
		joblib.dump(score_train, FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_trainFisher.pkl')
		joblib.dump(score_test , FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_testFisher.pkl')
		###
		print 'execution time:{:.2f}s'.format(T.time()-tStart)
#==============================================================================
#		tStart = T.time()
# 		# Naive Bayesian on Gaussian
# 		print 'FV_{}\tmix={}\tGaussian Naive Bayesian'.format(FV,MIXTURE)
# 		clf = GNB()
# 		clf.fit(fv_train, label_train)
# 		prob_train = np.delete(clf.predict_proba(fv_train),1, axis=1).flatten()
# 		prob_test  = np.delete(clf.predict_proba(fv_test) ,1, axis=1).flatten()
# 		score_train = np.append(score_train, [prob_train], axis=0)
# 		score_test  = np.append(score_test , [prob_test] , axis=0)
# 		print 'execution time:{:.2f}s'.format(T.time()-tStart)
# 		tStart = T.time()
# 		# Naive Bayesian on Bernoulli
# 		print 'FV_{}\tmix={}\tBernoulli Naive Bayesian'.format(FV,MIXTURE)
# 		for ALPHA in [0.1,1,10]:
# 			clf = BNB(ALPHA)
# 			clf.fit(np.abs(fv_train), label_train)
# 			prob_train = np.delete(clf.predict_proba(fv_train),1, axis=1).flatten()
# 			prob_test  = np.delete(clf.predict_proba(fv_test) ,1, axis=1).flatten()
# 			score_train = np.append(score_train, [prob_train], axis=0)
# 			score_test  = np.append(score_test , [prob_test] , axis=0)
# 		print 'execution time:{:.2f}s'.format(T.time()-tStart)
#==============================================================================
		########## SGD
		tStart = T.time()
		M = 'SGD'
		print 'FV_{}\tmix={}\tSGD'.format(FV,MIXTURE)
		for ALPHA in [0.0001]:
			for i,PENALTY in enumerate(['l2','l1','elasticnet']):
				for LR in ['optimal']:#,'constant','invscaling']:
					model.append(FV+'_'+M+str(PENALTY))
					clf = SGD(penalty=PENALTY,alpha=ALPHA,learning_rate=LR)#, class_weight='balanced')
					clf.fit(fv_train, label_train)
					prob_train = clf.decision_function(fv_train)
					prob_test  = clf.decision_function(fv_test)
					if (i == 0):# and (FV=='gen'):
						score_train = [prob_train]
						score_test  = [prob_test]
					else:
						score_train = np.append(score_train, [prob_train], axis=0)
						score_test  = np.append(score_test , [prob_test] , axis=0)
		###
		score_train = np.transpose(score_train)
		score_test  = np.transpose(score_test)
		joblib.dump(score_train, FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_trainFisher.pkl')
		joblib.dump(score_test , FV_pth+'score/'+str(MIXTURE)+'_'+FV+'_'+M+'_testFisher.pkl')
		###
		print 'execution time:{:.2f}s'.format(T.time()-tStart)
		tStart = T.time()

	model = pd.DataFrame(model)
	score_train = np.transpose(score_train)
	score_test  = np.transpose(score_test)
	joblib.dump(score_train, FV_pth+'score/'+str(MIXTURE)+'_trainFisher.pkl')
	joblib.dump(score_test , FV_pth+'score/'+str(MIXTURE)+'_testFisher.pkl')
 
#==============================================================================
# START TESTING!!!
#==============================================================================
	# np.set_printoptions(precision=2,threshold=np.nan)
	# f = open('./test/m'+str(MIXTURE)+'_score.txt','w')
	# best_uar = 0
	# best_set = ''
	# best_sup = list()
	# # Feature Selection from 10% ~ 40%
	# for PERCENTILE in np.arange(10,50,10):
	# 	SELECTOR = SelectPercentile(f_classif, PERCENTILE)
	# 	selected_train = SELECTOR.fit_transform(score_train, label_train)
	# 	selected_test = SELECTOR.transform(score_test)
	# 	support = model[SELECTOR.get_support(indices=False)]
	# 	support = support.to_dict(orient='list')[0]

	# 	for C in [0.1,1]:
	# 		SVM = SVC(C,KERNEL)#, class_weight='balanced')
	# 		SVM.fit(selected_train, label_train)
	# 		result = SVM.predict(selected_test)

	# 		print 'select:{}\tC={}'.format(PERCENTILE, C)
	# 		print 'support:\n{}'.format(support)
	# 		f.write('select:{}\tC={}\n'.format(PERCENTILE, C))
	# 		f.write('support:\n{}\n'.format(support))
	# 		accu = accuracy_score(label_test, result)*100
	# 		uar = recall_score(label_test, result, average='macro')*100
	# 		cm = confusion_matrix(label_test, result, labels=[True, False])
	# 		cm_norm = cm.astype('float') / cm.sum(axis=1)

	# 		if uar > best_uar:
	# 			best_uar = uar
	# 			best_set = 'select:{}\tC={}'.format(PERCENTILE, C)
	# 			best_sup = support
	# 		print 'acc: {:.4f}\t uar: {:.4f}\n cm:\n{}'.format(accu, uar, cm_norm)
	# 		f.write('acc: {:.4f}\t uar: {:.4f}\n cm:\n{}\n\n'.format(accu, uar, cm))
	# 	f.write('\n\n')
	# f.write('Best UAR: {:.4f} on\n{}\nwith support:\n{}'.format(best_uar,best_set,best_sup))
	# f.close()

print (T.time() - tStart)
