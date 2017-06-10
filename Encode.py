# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:05:06 2016

@author: John Li

Edit: Huiting hong
2017-0225

Encoded way - BOW(kmeans -> Hist.)
Input : signal features (processed by opensmile)
Output : Histograms 
"""
import os
import numpy as np
import glob
import random
import csv
import time
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def Partial_read(IDX,path,ary):
	# According to fID_ary, only read the # of features we need from the INPUT.
	features = []
	# IDX = 3
	if IDX == 0:
		s = 0
		e = ary[IDX]
		f = pd.read_table(path,sep=';',skiprows = 1,nrows = e-s,header = None)
	else:
		s = ary[IDX-1]
		e = ary[IDX]
		f = pd.read_table(path,sep=';',skiprows = s,nrows = e-s)

	name = f.values[0][0]
	fname = name
	for i in range(e-s):
		# print(f.values[i])
		features.append([float(j) for j in f.values[i][1:]])
	# print(features)
	
	return features, fname
	# printf(f.loc[[2]])
	# print(f.loc["train_0002.wav'"])

def fID(path):
	# Read first column to get file name.
	f = pd.read_table(path,sep=';',skiprows = 0,usecols = [0]) #AUTO SKIP THE FIRST ROW
	val = f.values
	# print(val)

	# Create fID_ary : calculate the # of each file name, store each start index and end index.
	fID_ary = []
	for i, ele in enumerate(val):
		if i == 0 :
			flag = ele
		elif ele != flag :
			flag = ele
			fID_ary.append(i)
	fID_ary.append(len(val))
	# print(fID_ary)
	return fID_ary

class BagOfWords:
    ''' Usage
    k = k clusters of kmeans
    B = BagOfWords(k)
    Voc = B.train(Samples)
    words, hist = B.makeHistogram(features)
    '''
    classifier = None

    def __init__(self, k):
        self.k = k
        self.classifier = KMeans(n_clusters=k, precompute_distances=True)  # init_size

    def train(self, features):
        # Kmeans into k clusters
        self.classifier.fit(features)
        voc = self.classifier.cluster_centers_  # the centers as vocabulary
        return voc

    def makeHistogram(self, feature):
        descriptors = feature[0]
        for descriptor in feature[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        if self.classifier == None:
            raise Exception("You need train this instance.")
        words = self.classifier.predict(descriptors)
        hist = np.bincount(words, minlength=self.k)
        return words, hist

class BOW(object):
    voc = []
    words = []
    hist = []


