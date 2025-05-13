**EEG Project Progress Report**

**INTRODUCTION**
Sleep staging from electroencephalography (EEG) is essential for diagnosing sleep disorders and advancing research on circadian rhythms, as well as for the development of wearable sleep‐tracking devices. While manual polysomnography (PSG) remains the gold standard, its reliance on multimodal data (EEG, electrooculography, electromyography, electrocardiography, and respiratory sensors) and labor-intensive scoring make it impractical for large-scale or real-time applications. Automated methods exist but often depend on predefined or assumed in-bed intervals. These methods, although highly sensitive for detecting sleep epochs, typically exhibit low specificity for wake epochs—especially problematic when the subtle frequency differences between relaxed wakefulness and light (N1) sleep are masked by noise. Our project aims to overcome these limitations by optimizing the classification of sleep stages from EEG data, with a particular focus on accurately detecting sleep onset and differentiating wakefulness from sleep in both daytime and nighttime patterns. An added objective is to explore whether the extracted features might serve as meaningful biomarkers for sleep onset and circadian regulation. Additionally, we are going to use two datasets to validate our results : MESA (classic dataset used for comparison) and ANPHY (smaller dataset that we are starting with).  

**PROBLEM DESCRIPTION**
Our project intends to optimize sleep stage classification by focusing on the identification of sleep onset. In practice, current methods—primarily designed for in-bed intervals—tend to overclassify sleep due to the predominance of sleep epochs and the similarity in frequency spectra between relaxed wakefulness and N1 sleep. This bias leads to an underestimation of wake time, a significant concern for patients with low sleep efficiency. Moreover, many existing algorithms require explicit prompting of wake and sleep times, which limits their generalizability. By directly extracting features from EEG data on a per-epoch (30-second) basis, we seek to develop classifiers that can operate without such constraints and that may reveal latent biomarkers associated with sleep transitions.

**INITIAL PIPELINE**
Our initial pipeline is constructed using EEG data from 29 subjects, each recorded with approximately 93 channels in EDF format. Sleep stage labels are provided as text files with one label per 30-second epoch. The pipeline employs a custom data loader (ANPDataLoader) that extracts key metadata (sampling rates, channel labels, and recording durations) and segments the data into 30-second epochs. Two feature extraction approaches have been implemented. Initially we wanted to do a power spectral density (PSD) method uses a Butterworth bandpass filter (0.5–40 Hz) and Welch’s method to compute relative band power in the delta, theta, alpha, and beta bands for each electrode. Next, a catch22-based method (via the pycatch22 package) computes 22 canonical time-series features that capture temporal autocorrelation, entropy, and distributional properties. Both approaches are designed to yield one feature vector per electrode per 30-second epoch, thereby aligning with the ground-truth sleep labels. However, we are actually now working on using transformers in general for this problem since that has not been done on this type of problem in the field and may lead to more nuanced insights (different than the other methods of feature exreaction) and we want to see how it might be able to outperform other models. 

**3.1 Mathematical Formulation of the Optimization Problem**

**Problem Setup:**
Given an EEG time series \( Y \) with associated covariates \( Z \), we seek to predict the sleep stage \( S \) for future time steps using a learned function:

$$ \hat{S}_t = f_{\theta}(Y_{t-l:t}, Z_{t-l:t+h}) $$

**Objective Function:**
We train \( f_{\theta} \) to maximize the log-likelihood of the predicted sleep stage distribution:

$$ \max_{\theta} \mathbb{E}_{(Y,Z) \sim p(D)} \mathbb{E}_{(t,l,h) \sim p(T \mid D)} \log p(S_{t:t+h} \mid \hat{\phi}) $$

where \( \hat{\phi} \) are the learned parameters of the predictive distribution.

**3.2 Implementation using MOIRA**

**MOIRA: Masked Encoder-based Universal Time Series Forecasting Transformer**

**Why MOIRA?**
- Handles multivariate time series effectively using a masked encoder architecture.
- Learns from unobserved data (zero-shot learning) to improve generalizability.
- Uses multi-patch projections for capturing hierarchical temporal structures.
- Outputs probabilistic distributions rather than single-point predictions.

**MOIRA-Based Classification Pipeline**

- **Preprocess EEG Data**
  - Load EDF recordings and segment into 5-second windows (this is also a challenge because the labels only occur once every 30 seconds).
  - Apply bandpass filtering (0.5–40 Hz).
  - Extract PSD features + Catch22 features for each epoch.

- **Feature Transformation for MOIRA**
  - Apply multi-patch projection layers to generate vector embeddings.
  - Use learnable masked embeddings to encode future sleep stage predictions.

- **Training Objective**
  - Train the model using cross-entropy loss:

$$ \text{loss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_i^k \log \hat{y}_i^k $$

where:
- \( K \) is the number of sleep stages (at least wake and N1).
- \( y_i^k \) is the predicted probability of sample \( i \) belonging to class \( k \).

- **Evaluation Metrics**
  - Accuracy of wake vs. N1 classification.
  - Precision & recall to balance specificity and sensitivity.
  - Cross-validation across subjects to ensure generalizability.

**4. What We've Learned So Far**
- We have started to understand what models our goals necessitate. Time-series forecasting models like the Google timesFM approach do not fit our classification problem since they use decoders and an encoder is much more fitting to the problem we are working on. 
- We learned about the MOIRA model's masked encoder approach and how it effectively captures temporal dependencies, which is applicable to our problem.
- We now understand catch22 on a deeper level and think we can go back to our initial idea of using it together with the MOIRA model.
- Additionally, we learned processing full 24-hour EEG recordings is currently infeasible without significant optimization or additional hardware (e.g., GPUs). We expect data aggregation to be necessary in completing this project.
- We will need to determine the optimal downsampled sampling rate that does not result in significant loss of information.

**FUTURE WORK**
Future efforts will focus on both computational optimization and methodological enhancement. In particular, 

- **Computational Optimization:**
   - Implement parallel processing (using joblib or Python’s multiprocessing) to reduce processing times.
   - Work on using lightning py and only train the last couple layers (not the middle to make it more efficient). 
   - Only work on one patient for next week.
   - Explore memory-efficient data loading techniques such as chunked processing and streaming.

- **Methodological Improvements:**
   - Reassess model architecture for potential memory optimization.
   - Investigate deep learning architectures such as temporal convolutional networks (TCNs) and Time-Series Transformers.

- **Cross-Domain Insights:**
   - Consult resources on financial time-series data processing, as similar challenges might provide insights into managing EEG datasets.

**CHALLENGES AND CONCERNS:**
- The complexity of EEG data and computational constraints have made model implementation more time-consuming than anticipated.
- The lack of consistent data modalities to the AI model has also been really challenging because conversion is not easy.
- We're struggling with downsampling the EDF files that we have because of Nyquist and also making sure that the powerband is the right data (not introducing standard downsampling but using basis functions).

**PROJECT ADJUSTMENT REQUEST:**
Given the difficulties encountered with the EEG dataset and the associated computational challenges, we are considering a shift in the project focus. We propose exploring time-series pattern detection techniques that are more aligned with computer science and statistical modeling practices, including trying the MOIRA neural network on existing datasets which are much better formatted for the problem (csv rather than the long edf data while we figure out downsampling). This pivot would maintain the core learning objectives while applying more universally relevant methodologies. We believe that a refined focus on time-series event detection could better connect with real-world applications, including in financial time-series analysis.

**CONCLUSION:**
While progress has been slower than desired, we have laid important groundwork in model selection and data preprocessing. Next steps involve optimizing data handling, exploring insights from related domains, applying more memory-efficient modeling strategies, and potentially shifting focus to time-series pattern detection.

**SOURCES:**
- [Dataset of 29 subjects](https://osf.io/r26fh/)
- [Catch22 Python documentation](https://time-series-features.gitbook.io/catch22/language-specific-docs/python)
- [Actigraphy paper](https://www.sciencedirect.com/science/article/pii/S2352721823001341?via%3Dihub)
- [Catch22 paper](https://link.springer.com/article/10.1007/s10618-019-00647-x)
- [MOIRA paper](https://arxiv.org/abs/2402.02592)