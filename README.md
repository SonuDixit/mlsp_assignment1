# mlsp_assignment1
4. Fischer faces - Data is posted here
15 subject faces with happy/sad emotion are provided in the data. Each image is of
100x100 matrix. Perform PCA on to reduce the dimension from 10000 to K (using
PCA for high dimensional data) and then perform LDA to one dimension. Plot the one
dimension features for each image. Select the optimum threshold to classify the emotion
and report the classification accuracy on the test data. What is the best choice of K
which gives the maximum separability ?

5. Speech spectrogram - We have clean and noisy speech files here
The files are in wav format sampled at 16kHz. Write a function to compute the spectrogram
of clean and noisy files (use 25 ms triangular window with a shift of 10 ms, compute
256 point magnitude FFT and retain the first 128 dimensions in each window, apply log
of the magnitude of the FFT.). For example, a speech file of 3s will have a spectrogram of
size 128×298 (without any padding at the end, where 128 is the dimension of the feature
vector and 298 is the number of frames).
(a) Assume that each speech frame (256 dimensional vector) is independent of each other.
From the clean files, compute the whitening transform. Apply the transform on the
noisy speech features. Find the average of the absolute value of the non-diagonal
entries of the sampled covariance matrix of the “whitenned” clean and noisy speech
features. Comment on use of whitenning when the data distribution has changed.
(b) Repeat the above procedure by reversing the roles of clean and noisy files.
