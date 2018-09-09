import numpy as np
from scipy.io import wavfile
path=".\\Data\\speechFiles\\"
#fs, data = wavfile.read(path + "noise.wav")
##apply triangular..window size 25ms, slide length 10ms
def prepare_data_applying_trian_window(data_1d,window_length=400,shift=160):
    data_1d = np.asarray(data_1d)
#    print(data_1d.shape[0])
    total_data_points = 1 + int((data_1d.shape[0]-window_length) /shift)
#    print(total_data_points)
    data = np.zeros(shape=(total_data_points,window_length))
    window_to_mult = np.bartlett(window_length+2).tolist()
    window_to_mult.remove(0)
    window_to_mult.remove(0)
    window_to_mult = np.asarray(window_to_mult)
#    print(window_to_mult)
    for i in range(total_data_points):
        data[i,:] = data_1d[i*shift : i*shift + window_length] * window_to_mult
    return data

def apply_fft(data,output_len=128):
    fft_data = np.zeros(shape=(data.shape[0],128))
    for i in range(data.shape[0]):
        a = np.fft.rfft(data[i,:],n=256)
        fft_data[i] = np.delete(a,-1)
    return fft_data

def whitening(data):
    #data is nXd
    # returned data format is same
    data = np.asarray(data)
    data = data -np.mean(data,axis=0)
    cov = np.matmul(data.T,data)
    eig_val,eig_vec = np.linalg.eig(cov)
    whitened_data = np.matmul(np.matmul(eig_vec,np.sqrt(np.linalg.inv(np.diag(eig_val)))).T,data.T)
    whitened_data = whitened_data.T
    return whitened_data, np.matmul(eig_vec,np.sqrt(np.linalg.inv(np.diag(eig_val)))).T
    
def get_cov(data):
    means = np.mean(data,axis=0)
    return np.matmul((data-means).T,data-means)
def get_sum_nonzero_nondiag(mat):
    s=0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                if mat[i][j] != 0:
                    s += np.abs(mat[i][j])
    return s
def part_a():
    """From the clean files, compute the whitening transform. 
    Apply the transform on the noisy speech features. 
    """
    fs, data = wavfile.read(path + "clean.wav")
    data = prepare_data_applying_trian_window(data)
    data = apply_fft(data)
    means = np.mean(data,axis=0)
    w_data, mat_to_mult = whitening(data)
    
    fs, noisy_data = wavfile.read(path + "noisy.wav")
    noisy_data = prepare_data_applying_trian_window(noisy_data)
    noisy_data = apply_fft(noisy_data)
    ####applying the transformation
    noisy_data = noisy_data - means
    whitened_noisy_data = np.matmul(mat_to_mult,noisy_data.T).T
    #get cov matrix of this data
    cov_whiten = get_cov(whitened_noisy_data)
    sum_nonzero = get_sum_nonzero_nondiag(cov_whiten)
    print(sum_nonzero/(whitened_noisy_data.shape[0] * (whitened_noisy_data.shape[0]-1)))

def part_b():
    """From the noisy files, compute the whitening transform. 
    Apply the transform on the clean speech features. 
    """
    fs, data = wavfile.read(path + "noisy.wav")
    data = prepare_data_applying_trian_window(data)
    data = apply_fft(data)
    means = np.mean(data,axis=0)
    w_data, mat_to_mult = whitening(data)
    
    fs, noisy_data = wavfile.read(path + "clean.wav")
    noisy_data = prepare_data_applying_trian_window(noisy_data)
    noisy_data = apply_fft(noisy_data)
    ####applying the transformation
    noisy_data = noisy_data - means
    whitened_noisy_data = np.matmul(mat_to_mult,noisy_data.T).T
    #get cov matrix of this data
    cov_whiten = get_cov(whitened_noisy_data)
    sum_nonzero = get_sum_nonzero_nondiag(cov_whiten)
    print(sum_nonzero/(whitened_noisy_data.shape[0] * (whitened_noisy_data.shape[0]-1)))
    
part_a()
part_b()

#print(prepare_data_applying_trian_window([1,2,3,4,5],3,2))