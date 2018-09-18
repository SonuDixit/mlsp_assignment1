
import numpy as np
def pca(data,num_components):
    #data is mean 0 data.
    ##data is in num_example X num_features
    #num features  >>> num examples
    #data doesnt have first column showing feature names
    #class labels have been removed from data
    data = np.asarray(data)
    num_examples = data.shape[0]
    num_features = data.shape[1]
    mean_for_pca = np.mean(data,axis=0)
    data = data - mean_for_pca
    if num_examples < num_features and num_components<=num_examples:
        """
        calculate data X data.T
        get its eigen_vals and eig vecs
        """
        cov_mat = np.matmul(data,data.T)
        eig_val,eig_vec = np.linalg.eig(cov_mat)
        indices = np.argsort(eig_val)
        eig_vec_forpca = np.zeros(shape = (num_features,num_components))
        for i in range(num_components):
#            print("eval =",eig_val[indices[-(i+1)]])
            eig_vec_forpca[:,i] = np.matmul(data.T,eig_vec[:,indices[-(i+1)]])
#            print("norm is",np.linalg.norm(eig_vec_forpca[:,i]))
            eig_vec_forpca[:,i] = eig_vec_forpca[:,i]/np.sqrt(np.abs(eig_val[indices[-(i+1)]]))
#            print("norm is",np.linalg.norm(eig_vec_forpca[:,i]))
#        for i in range(eig_vec_forpca.shape[1]):
#            print("norm is",np.linalg.norm(eig_vec_forpca[:,i]))
        return eig_vec_forpca,mean_for_pca,np.matmul(data,eig_vec_forpca)
    else:
        if num_components < num_features:
            print("implement conventional pca")
            cov_mat = np.matmul(data.T,data)
            eig_val,eig_vec = np.linalg.eig(cov_mat)
            print("......worked")
            indices = np.argsort(eig_val)
            eigvec2= np.zeroes(eig_vec.shape[0],num_components)
            for i in range(num_components):
                eigvec2[:,i] = eig_vec[:,-i-1]
            return eigvec2,np.matmul(data,eigvec2)
        else:
            return("cant do...its dimensionality increasing...")


#evec,data = pca(np.ones(shape=(2,5)),num_components=2)
