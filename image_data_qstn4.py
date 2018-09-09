# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:44:26 2018

@author: vissim
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from pca_for_large_dimension import pca
from lda_to_1dim import LDA_to_1dim
import numpy as np
import os
path_train=".\\Data\\emotion_classification\\train\\"
path_test =".\\Data\\emotion_classification\\test\\"

def read_data(dir_path,happy_label=1,sad_label=0):
    #path=".\\Data\\emotion_classification\\train\\"  ....an example path
    #notice "\\" and dot usage
    #dir_path is the directory path
    #all files will be read from this directory
    files = os.listdir(dir_path)
#    print(files)
    all_data =[]
    label=[]
    for file_name in files:
        im = imread(dir_path + file_name)
        arr = np.asarray(im)
        arr= np.reshape(arr,newshape=(1,arr.shape[0] * arr.shape[1]))
        all_data.append(arr)
        
        if "happy" in file_name :
            label.append(1)
        else:
            label.append(0)
    all_data = np.asarray(all_data) #converted to 3d
    all_data = np.reshape(all_data,newshape = (-1,all_data.shape[2])) ##assuming image data
    
    return all_data,label

def cal_accuracy(pred,actual):
    correct = 0
    print(len(pred) == len(actual))
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            correct+=1
    return correct/len(pred)

def train(path_train,k):
    ##it should return threshold,eig_vecs,lda_vec,mean_for_pca
    data,label = read_data(path_train)
    eig_vec, mean_for_pca, low_dim_data = pca(data,k)
    one_dim_data,lda_vec = LDA_to_1dim(low_dim_data,label)
    plt.scatter(one_dim_data.real.tolist(),label,c=label)
    plt.show()
    ##calculate threshold
    sum_0=[]
    sum_1=[]
    for i in range(len(label)):
        if label[i]== 0:
            sum_0.append(one_dim_data[i])
        else:
            sum_1.append(one_dim_data[i])
    threshold = (np.mean(sum_0) + np.mean(sum_1))/2
    flag = False
    if threshold > np.mean(sum_0):
        flag = True
    return eig_vec,mean_for_pca,lda_vec,threshold,flag

###testing..
def test(path_test,eig_vec,mean_for_pca,lda_vec,threshold,flag):
    data,label = read_data(path_test)
    data = data-mean_for_pca
    pca_data = np.matmul(data,eig_vec)
    one_dim_data = np.matmul(pca_data,lda_vec)
    pred_label =[]
    if flag:
        for i in range(len(one_dim_data)):
            if(one_dim_data[i] < threshold):
                pred_label.append(0)
            else:
                pred_label.append(1)
    else:
        for i in range(len(one_dim_data)):
            if(one_dim_data[i] < threshold):
                pred_label.append(1)
            else:
                pred_label.append(0)
    print("predictions are",pred_label)
    print("actuals are",label)
    print(cal_accuracy(pred_label,label))


for i in [8,10,12,13,14,15,16,17,18,19,20]:
    print("i is",i)
    eig_vecs,mean_for_pca,lda_vec,threshold,flag=train(path_train,i)
    test(path_test,eig_vecs,mean_for_pca,lda_vec,threshold,flag)
#eig_vecs,lda_vec,threshold=train(path_train,10)
#test(path_test,eig_vecs,lda_vec,threshold)