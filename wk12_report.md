# Sleep Stage Classification: Deep Learning Approaches for N1 Detection

## Problem Definition

### Problem Statement
Our project aims to improve automated sleep stage classification from EEG data, with particular emphasis on accurately detecting the transition from wakefulness to sleep (N1 stage).

### Motivation
Sleep disorders affect approximately 70 million Americans annually, and accurate sleep staging is essential for diagnosis and treatment. Manual scoring by experts is time‑consuming and inconsistent (κ ≈ 0.70–0.80 between human scorers), with N1 being the most challenging stage to identify due to its transitional nature and subtle EEG changes. Automated classification can accelerate sleep studies analysis, enable large‑scale sleep research, and improve home‑based sleep monitoring.

### Success Metrics
- **Primary**: F1‑score for N1 classification, targeting > 0.60  
- **Secondary**:  
  - Overall accuracy > 85 %  
  - Cohen’s Kappa > 0.75 compared to expert annotations  

## Technical Approach

### Literature Review
- **Traditional Methods**  
  - Spectral power analysis and time‑series feature extraction  
  - YASA (Yet Another Sleep Algorithm) for spindles and slow waves  
  - Rule‑based systems using domain knowledge of sleep architecture  
- **Deep Learning Approaches**  
  - **DeepSleepNet** (CNN‑based): 85 % overall accuracy, N1 F1 ≈ 0.35  
  - **U‑Sleep** (Transformer‑based): 88 % overall accuracy, N1 F1 ≈ 0.42  
  - **SeqSleepNet**: Attention mechanisms to model temporal dependencies  

**Challenges with N1**  
- Transitional nature (1–5 min duration)  
- Overlap with wakefulness (α) and early N2 (θ) patterns  
- Severe class imbalance (5–10 % of recording)  
- Low inter‑scorer reliability (κ < 0.7)  

## Modeling Approach

### Mathematical Formulation
Given sequences of epochs  
$$
\{x_1, x_2, \dots, x_T\},\quad \{y_1, y_2, \dots, y_T\},\quad T = 20
$$  
we minimize the average cross‑entropy loss:  
$$
\min_{\theta}
\sum_{(X,Y)\in \mathcal{D}}
\left[
  -\frac{1}{T}
  \sum_{t=1}^{T}
  \sum_{k=1}^{5}
  \mathbb{I}\{y_t = k\}\,\log p_{\theta}(k\mid x_t)
\right]
$$  
where \(p_{\theta}(k\mid x_t)\) is the model’s predicted probability for class \(k\) at epoch \(t\).

To address class imbalance, we adopt **focal loss** with extra weight on N1:  
$$
\mathcal{L}(p_t)
= -\alpha_t\,(1 - p_t)^{\gamma}\,\log(p_t)
$$  
with  
- \(\alpha_t = 0.75\) for N1, \(\alpha_t = 0.25\) for others  
- \(\gamma=2\) focusing parameter  

### Optimization Choices
- **Optimizer**: Adam (adaptive LR, momentum)  
- **LR Scheduler**: ReduceLROnPlateau (factor 0.5, patience 3)  
- **Gradient Clipping**: Norm clips at 1.0  

### Hyperparameter Tuning
- \(\alpha_{\mathrm{N1}}\in\{0.5,0.75,0.9\}\), \(\gamma\in\{1,2,3\}\)  
- Transformer layers: 1–3; heads: 2–8  
- Sequence length: 10, 20, 30 epochs  
- Batch size: 8–32  

## Implementation Details

- **Data Processing**: MNE for EEG loading, filtering, epoching, annotation mapping  
- **Sequence Creation**: Sliding windows of 20 epochs with stride of 10  
- **Model Architecture**:  
  - **EpochEncoder**: CNN stack ➔ 128‑d embedding  
  - **Positional encoding**: learned vectors  
  - **TransformerEncoder**: 2 layers, 4 heads  
  - **Classifier**: linear to 5 classes  
- **Post‑processing**: median smoothing (kernel = 5) for temporal consistency  
- **Train/Test Split**: true subject‑based (no subject leakage)

## Limits Encountered
- **Compute**: M1 MacBook (8 GB RAM, no GPU)  
- **Data Leakage**: fixed with true‑subject split  
- **Class Imbalance**: only ~ 5 % N1 epochs  

## Results

### Epoch 35/35 Detailed Metrics (Pre‑smoothing)
| Class           | Precision | Recall | F1‑score | Support  |
|:---------------:|----------:|-------:|---------:|---------:|
| Wake (W)        |      0.99 |   0.92 |    0.96 |  110 105 |
| N1              |      0.30 |   0.57 |    0.39 |    7 996 |
| N2              |      0.85 |   0.70 |    0.77 |   33 877 |
| N3              |      0.71 |   0.82 |    0.76 |    8 882 |
| REM             |      0.64 |   0.88 |    0.74 |   13 160 |
| **Accuracy**    |           |        |    0.85 |  174 020 |
| **Macro avg**   |      0.70 |   0.78 |    0.72 |  174 020 |
| **Weighted avg**|      0.89 |   0.85 |    0.87 |  174 020 |

### Final Metrics (with smoothing)
| Class           | Precision | Recall | F1‑score | Support  |
|:---------------:|----------:|-------:|---------:|---------:|
| Wake (W)        |      0.99 |   0.95 |    0.97 |  110 105 |
| N1              |      0.39 |   0.30 |    0.34 |    7 996 |
| N2              |      0.81 |   0.79 |    0.80 |   33 877 |
| N3              |      0.71 |   0.79 |    0.75 |    8 882 |
| REM             |      0.63 |   0.89 |    0.74 |   13 160 |
| **Accuracy**    |           |        |    0.88 |  174 020 |
| **Macro avg**   |      0.71 |   0.74 |    0.72 |  174 020 |
| **Weighted avg**|      0.88 |   0.88 |    0.88 |  174 020 |

## Interpretation
- **N1**: F1 = 0.39 (pre‑smoothing), drops to 0.34 after smoothing  
- **Wake, N2, REM**: ≥ 82 % F1 post‑smoothing  
- **N3**: 75 % F1 (some confusion with N2)  

## Progress vs. Expectations
- **Goal**: N1 F1 > 0.50  
- **Achieved**: N1 F1 = 0.39 (pre‑smoothing), 0.34 (post‑smoothing)  
- **Trade‑offs**: smoothing improves overall consistency but reduces N1 recall  

## Future Work
- HMM/Markov layers for transition constraints  
- Self‑supervised pretraining on unlabeled EEG  
- Subject adaptation via fine‑tuning  
- Cross‑dataset validation (MASS, SHHS)  

## Conclusion
Our CNN + Transformer model with N1‑focused focal loss yields a peak N1 F1 = 0.39 before smoothing and 0.34 after smoothing (with balanced focal loss), while achieving 88 % overall accuracy. Further work on temporal modeling and data augmentation is needed to boost N1 detection in real‑world settings.
