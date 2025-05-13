Downsampling:
our F_s is 1000Hz, so Nyquist is Fs/2 = 500Hz -> to accuratey capture all information in the original signal, there shouldn't be any freq components above 500Hz

firts, we use LFP to remove high-freq components from the signal that we don't need/can't represent with a lower sampling rate

filtering doesn't change the fact that the raw data has Nyq. limit 500; it alters the effective BW (bandwidth)of the signal (by discarding frequencies above a chosen cutoff)

Downsampling/decimation step:
after filtering, we purposely reduce the number of samples/lower the samplign rate, but the new sam_rat F'_s still must be at least twice the highest frequency present in the filtered signal

Filtering reduces the bandwidth of the signal by removing high frequencies. After filtering, the effective maximum frequency component of the signal is lower. Therefore, when downsampling, the new Nyquist frequency based on the new, lower sampling rate—not the original one.

Wavelet transform:
basis functions are scaled and shifted versions of a single mother wavelet

apply discrete wavelet transform (DWT) to decompose the signal into wl coefficients
keep only the largest-magnitude coefficients (corresponding to the largest eigenvalues if you apply further dimensionality reduction)

for dim red:
you could treat the F/WL coefficients (or even the TS data over a sliding window) as a feature vector and identify main modes/directions via PCA/SVD

project to lower-dim basis

BIO stuff:
delta, theta, alpha, and beta bands are all below 40 Hz -> we could filter out high-freq content and safely reduce the sampling rate
but in sleep research, very common choice is 250 Hz - high enough to capture detail for events like sleep spindles (~12-16 Hz) while reducing data size

reasonable parameters:
HP filter with cutoff frequency around 0.3-0.5 Hz -> removes slow drifts and very low-fr artifacts
LP filt with cutoff ~30-40 Hz -> eliminates high-fr noise (e.g. muscle artifacts)
then the effective signal bandwidth becomes 40 Hz so Nyquist is 80 Hz to avoid aliasing


notch for power line interference - check paper to see if they did it