at /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results:
Each saved file is a NumPy array containing the PCA-reduced wavelet energy features.

computed via wavelet_pca.py script, which does the following:
1) wavelet energy features:
- For each EEG epoch (and for each channel in that epoch), compute the discrete wavelet transform using a specified wavelet (here, the default is 'db4').
- Calculate the energy (sum of squares) for both the approximation and detail coefficients at various decomposition levels.
- These energy values form a feature vector that represents the time–frequency characteristics of the EEG signal.

2) PCA dimensionality reduction:
- Apply PCA to reduce the dimensionality of the feature vectors (capture 95% of the variance explained, quite strict)
- Save the reduced features in a separate NumPy array.

cur

