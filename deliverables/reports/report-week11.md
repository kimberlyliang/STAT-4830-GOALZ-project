# REPORT: Transitioning to Sleep-EDF and Our Current Model Architecture

## INTRODUCTION
Our project aims to improve sleep stage classification from EEG data, with particular emphasis on accurately detecting the transition from wakefulness to N1. While previous work with in-bed intervals and large electrode arrays yielded promising overall accuracy, it struggles to identify N1 stage, which corresponds to between-state transitions. In light of these limitations, we have **shifted our data source** to the publicly available Sleep-EDF dataset. This move allows us to benchmark against established baselines, exploit a well-documented labeling scheme, and facilitate broader comparisons with state-of-the-art methods published in the literature.

## SHIFT TO SLEEP-EDF
Our previous manual feature extraction pipeline computed a set of 47 spectral and time-series features which were subsequently used for supervised learning on the Anphy dataset consisting of EEG recordings of 29 subjects. Basic classifier models such as random forests or XGboost yielded high accuracy (ROC >0.95) for N3 stage, fairly good performance for N2, wakefulness, and REM sleep, and poor performance for N2 (~.06), with high numbers of false positives and false negatives across different hyperparameter settings. 

Upon additional literature review, we have decided to switch to a new dataset, Sleep-EDF, as it requires less storage per subject and has been validated on several models.

The Sleep-EDF database contains overnight polysomnography (PSG) recordings, including EEG, EOG, and chin EMG, sampled at or resampled to 100 Hz. Crucially, it includes consistent 30-second annotations for the canonical sleep stages (W, N1, N2, N3, REM). We selected **Fpz-Cz** as our primary EEG channel (plus EOG and EMG when needed) to preserve essential signals for detecting light sleep. This single-channel or multi-channel flexibility allows us to explore the effects of sparse electrode configurations on classification performance.

## DATA PREPROCESSING PIPELINE
### Data Source and Channel Selection
- **Dataset:** Sleep‑EDF (both Cassette and Telemetry subsets) containing PSG recordings with corresponding hypnogram annotations.
- **Channels:**  
  - Primary EEG: "EEG Fpz‑Cz" (selected as the best single EEG channel for capturing N1 characteristics)  
  - EOG: "EOG horizontal"  
  - Optionally, chin EMG may be included when multi-channel processing is desired (but not implemented in this analysis)
  
Our pipeline is modular and allows switching between single‑channel (EEG only) and multi‑channel (EEG+EOG+EMG) configurations, which is important for balancing computational efficiency on an M1 MacBook with 8 GB of RAM.

### Preprocessing Steps
1. **Loading and Resampling:**  
   Using MNE, each PSG EDF file and its corresponding hypnogram file are loaded with the selected channels. Data are resampled to 100 Hz to preserve the frequency range up to approximately 90 Hz while keeping the computational load manageable.

2. **Filtering:**  
   A 0.5–30 Hz bandpass filter is applied to EEG and EOG channels to remove DC drift and high-frequency noise. (For EMG, different or no filtering is applied as appropriate.)

3. **Epoching and Normalization:**  
   The recordings are segmented into 30‑second epochs (3000 samples per epoch). Each epoch is normalized (e.g., z-scored) to reduce inter-subject variability.

4. **Saving Processed Data:**  
   For each recording, the processed epochs (with shape (n_epochs, n_channels, 3000)) and corresponding numeric labels (mapped as 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM) are saved as compressed .npz files. This format is compatible with both PyTorch and TensorFlow for subsequent deep learning.

## MODEL ARCHITECTURE AND TRAINING PIPELINE
### Overview
Our deep learning model is designed to capture both local spectral features and long-range temporal context. The architecture combines a CNN-based epoch encoder with a Transformer-based sequence model.

### CNN Epoch Encoder
- **Input:** Each 30‑second epoch (shape: (n_channels, 3000)); for single‑channel, n_channels = 1; for multi‑channel, n_channels is 2 or 3.
- **Operation:** A series of 1D convolution and pooling layers extract a 128‑dimensional embedding that captures the time-frequency features of the epoch.

### Transformer Sequence Model
- **Input:** A sequence of CNN embeddings, formed by grouping consecutive epochs (e.g., 20 epochs per sequence).
- **Operation:** A Transformer encoder with self‑attention layers processes these embeddings, capturing temporal dependencies and transition dynamics that aid the identificaion of the ambiguous N1 stage.
- **Output:** For each epoch in the sequence, the model outputs a probability distribution over the five classes using a final linear classification layer with softmax activation.

### Training Details
- **Loss Function:** Multiclass cross‑entropy loss is minimized over all epochs in a sequence. Optionally, class weights (or focal loss) are applied to address the underrepresentation of N1.
- **Optimization:** We use the Adam optimizer with a moderate learning rate, training for 20 epochs on a subject-level split (70% train, 30% validation) to avoid data leakage.
- **Performance Metrics:**  
  - Training Accuracy: ~89.05%  
  - Validation Accuracy: ~86.21%  
  - Macro F1 Score: ~81.33%  
  - Cohen’s Kappa: ~0.7935  
  - Per-Class F1: Wake: 0.83, N1: 0.61, N2: 0.90, N3: 0.85, REM: 0.88
 

### Mathematical Formulation
For each sequence of epochs \(\{x_1, x_2, \dots, x_T\}\) with corresponding labels \(\{y_1, y_2, \dots, y_T\}\) (where \(T = 20\)), our optimization minimizes:

\[
\min_{\theta} \sum_{(X,Y)\in \mathcal{D}} \left[ -\frac{1}{T} \sum_{t=1}^{T} \sum_{k=1}^{5} \mathbb{I}\{y_t = k\} \log \,p_\theta(k \mid x_t) \right],
\]

where:
- \(p_\theta(k \mid x_t)\) is the predicted probability for class \(k\) at epoch \(t\),
- \(\theta\) represents all trainable parameters (CNN and Transformer weights), and
- \(\mathcal{D}\) is the training set.

## CURRENT RESULTS AND INTERPRETATION
Our CNN+Transformer model yields:
- **Train Loss:** 0.2754  
- **Train Accuracy:** 89.05%  
- **Validation Loss:** 0.3797  
- **Validation Accuracy:** 86.21%

The class-specific results indicate that while N2, N3, REM, and Wake are classified robustly, N1 (F1 ~0.61) remains the most challenging class. The small gap (~2.8%) between training and validation accuracy suggests minimal overfitting and confirms that the architecture is generalizing well. The Transformer’s ability to capture temporal context helps mitigate some ambiguity, though additional constraints (e.g., transition modeling) may further improve N1 detection.

## ARCHITECTURAL MODIFICATIONS AND PERFORMANCE IMPACT

### Multi-Scale CNN Epoch Encoder 
We replaced the original single-branch CNN encoder with a multi-scale design that processes each 30‑second epoch using parallel 1D convolutional branches with kernel sizes of 50, 100, and 150. The rationale was to capture rhythms at varying temporal resolutions, thereby extracting richer feature representations from the EEG signals. The outputs from these branches were concatenated and then mapped to a 128‑dimensional embedding via a fully connected layer.

### Class Weighting in the Loss Function
Recognizing the underrepresentation of the N1 stage, we modified the loss function by incorporating class weights. In our implementation, a higher weight was assigned to the N1 class (e.g., [1.0, 5.0, 1.0, 1.0, 1.0]) to penalize misclassifications of this critical transitional stage more heavily during training.

### Performance Impact
Despite the sound theoretical motivation, the introduction of the multi-scale encoder and class weighting led to a decrease in overall performance.
- Training Accuracy: ~79.03%  
  - Validation Accuracy: ~80.37%  
  - Macro F1 Score: ~53.74%  
  - Cohen’s Kappa: ~0.4878  
  - Per-Class F1: Wake: 0.25, N1: 0.36, N2: 0.78, N3: 0.73, REM: 0.54


## FUTURE WORK
1. **Transition Constraints:** Integrate a Markov or HMM layer to penalize unlikely transitions, further refining N1 detection.
2. **Self-Supervised Pretraining:** Pretrain the encoder on unlabeled EEG data to improve representations for rare classes.
3. **Channel Experiments:** Systematically evaluate single-channel versus multi-channel inputs to confirm the minimal set needed for high performance.
4. **Subject Adaptation:** Explore fine-tuning or domain adaptation strategies for subjects with fragmented sleep.
5. **Extended Evaluation:** Validate the model on additional recordings and potentially extend to cross-dataset generalization.

## CONCLUSION
By shifting to the Sleep-EDF dataset, we have aligned our work with a well-established benchmark and streamlined our preprocessing pipeline to efficiently extract 30-second epochs from selected channels (in particular EEG Fpz‑Cz and EOG horizontal). Notably, this shows that effective channel selection can eliminate redundance and substantially lessen  computational burden, without compromising accuracy. Our end-to-end CNN+Transformer model demonstrates robust overall performance (~86% validation accuracy) with only a modest gap to training performance. Although N1 remains the most challenging stage, our preliminary results suggest that further integration of transition dynamics and domain-specific pretraining may push performance beyond human-expert levels.
