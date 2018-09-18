
import numpy as np
def LDA_to_1dim(data,label):
    #calculate distance between mean in new coordinates
    # calculate within class scatter matrices.
    #data is assumend to be in num_exam x dim format
    # label is assumed to contain 1 and 0
    data = np.asarray(data)
    label = np.asarray(label)
    c1data_index = np.where(label == 0)
    c2data_index=np.where(label == 1)
    
    c1data = data[c1data_index]
    c2data = data[c2data_index]
    mean_c1 = np.matrix(np.mean(c1data,axis=0))
    print("shape of mean_c1 is",mean_c1.shape)
    mean_c2 = np.matrix(np.mean(c2data,axis=0))
    c1_cov = np.matmul((c1data-mean_c1).T, c1data-mean_c1)
    c2_cov = np.matmul((c2data-mean_c2).T,c2data-mean_c2)
    sw_inverse = np.linalg.inv(c1_cov + c2_cov)
    sb = np.matmul((mean_c1-mean_c2).T,mean_c1-mean_c2)
#    print(sb)
    print("shape of sb is",sb.shape)
    eig_val,eig_vec = np.linalg.eig(np.matmul(sw_inverse,sb))
    index = np.argsort(eig_val)
    reqd_vec = eig_vec[:,index[len(index)-1]]
    return np.matmul(data,reqd_vec),reqd_vec
    
#data=[[1,2,2],[1,1,1],[1,2,3]]
#label=[0,1,1]
#data = np.asarray(data)
#label = np.asarray(label)
#c1data_index = np.where(label == 0)
#c2data_index=np.where(label == 1)
#
#c1data = data[c1data_index]
#c2data = data[c2data_index]
