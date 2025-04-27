# Sleep Stage Classification: Deep Learning Approaches for N1 Detection

## Problem Definition

### Problem Statement
Automate sleep stage classification from scalp EEG, with a focus on accurately detecting the N1 (sleep‑onset) stage.

### Motivation  
Sleep disorders affect ~70 million Americans annually, and manual expert scoring is time‑consuming and inconsistent (κ ≈ 0.70–0.80) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}. Reliable N1 detection is critical for assessing sleep onset latency and overall sleep quality.

### Success Criteria & Measurable Outcome  
We will consider this project a success if, on held‑out test data:
- **N1 F1‑score ≥ 0.70**  
- **F1‑score ≥ 0.90 for Wake, N2, N3, and REM**  
- **Overall accuracy ≥ 90 %**  
- **Cohen’s Kappa ≥ 0.85**  

## Technical Approach

### Literature Review  
- **Feature‑based methods**: spectral power analysis; YASA for spindles & slow waves.  
- **DeepSleepNet (CNN)**: 85 % overall accuracy, N1 F1 ≈ 0.35 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}  
- **U‑Sleep (Transformer)**: 88 % overall accuracy, N1 F1 ≈ 0.42 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}  
- **SeqSleepNet**: self‑attention over epoch sequences.  

**Key challenge**: N1 is transient (1–5 min), overlaps wake/early N2, and is under‑represented (5–10 % of epochs).

## Modeling Approach

### Data & Preprocessing
- **Dataset**: Sleep‑EDF (Fpz‑Cz EEG; optionally EOG).  
- **Filtering**: bandpass 0.5–30 Hz; notch 60 Hz; downsample to 100 Hz.  
- **Epoching**: 30 s windows, sliding sequence of 20 epochs (stride = 10).  
- **Normalization**: per‑channel z‑score.

### Model Architecture
1. **Epoch Encoder**: three 1D conv layers (kernels 5→3→3), ReLU + max‑pool → flatten → FC to 128‑d embedding.  
2. **Positional Encoding**: learnable 20×128 tensor added to embeddings.  
3. **Transformer Encoder**: 2 layers, 4 heads, model dim = 128, feedforward = 512, dropout = 0.1.  
4. **Classifier**: linear projection → 5‑way softmax.  
5. **Post‑processing**: median filter (kernel = 5) for temporal smoothing.

### Mathematical Formulation
Let a sequence \(X=\{x_{t-19},\dots,x_t\}\) and labels \(Y=\{y_{t-19},\dots,y_t\}\). We minimize focal loss:
\[
\mathcal{L} = \sum_{i=1}^{T}
  -\alpha_{y_i}(1 - p_{i,y_i})^\gamma \log p_{i,y_i},
\]
where
\[
\alpha_{y_i} =
\begin{cases}
0.75,&y_i=\text{N1}\\
0.25,&\text{otherwise}
\end{cases}
,\quad
\gamma=2,
\]
and \(p_{i,k}\) is the model probability for class \(k\) at epoch \(i\).

## Optimization & Hyperparameter Tuning
- **Optimizer**: Adam (lr = 1×10⁻⁴; β₁=0.9, β₂=0.999)  
- **Scheduler**: ReduceLROnPlateau (factor 0.5, patience 3)  
- **Gradient clipping**: norm ≤ 1.0  
- **Sampling**: `WeightedRandomSampler` to rebalance classes  
- **Tuning procedure** (5‑fold CV):
  - \(\alpha_{\mathrm{N1}}\in\{0.5,0.75,0.9\}\)  
  - \(\gamma\in\{1,2,3\}\)  
  - Transformer layers: 1–3; heads: 2–8  
  - Sequence length: 10, 20, 30 epochs  
  - Batch size: 8–32  

## Implementation Details
- **Frameworks**: PyTorch Lightning, MNE.  
- **Hardware**: NVIDIA V100 GPU, 16 GB RAM.  
- **DataLoader**: `num_workers=4`; batch size = 16.  
- **Logging**: TensorBoard (loss & metric curves).  
- **Code practices**: subject‑wise train/test split (no leakage); checkpointing on validation N1 F1.

## Limits Encountered
- **Memory**: large sequence windows → reduced batch size.  
- **Class imbalance**: N1 ~ 5 % of data → focal loss + sampling.  
- **Compute time**: ~ 10 min/epoch, limited exhaustive sweeps.  
- **Missing labels**: ~1 % of epochs dropped due to absent hypnogram entries.

## Results

### Quantitative Performance

| Class       | Target F1 | Achieved F1 (post‑smooth) |
|:-----------:|:---------:|:-------------------------:|
| Wake (W)    | ≥ 0.90    | 0.97                      |
| **N1**      | ≥ 0.70    | 0.34                      |
| **N2**      | ≥ 0.90    | 0.80                      |
| **N3**      | ≥ 0.90    | 0.75                      |
| **REM**     | ≥ 0.90    | 0.74                      |
| **Overall** | ≥ 0.90    | 0.88                      |
| Cohen’s κ   | ≥ 0.85    | 0.78                      |

**Baseline Comparison**:

| Model        | Overall Acc. | N1 F1 |
|--------------|-------------:|------:|
| DeepSleepNet | 85 %         | 0.35 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7} |
| U‑Sleep      | 88 %         | 0.42 :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9} |
| **Ours**     | 88 %         | 0.34  |

### Visualizations
- **Training curves**: convergence of loss & N1 F1.  
- **Confusion matrix**: highlights N1↔Wake and N1↔N2 errors.  
- **Hypnogram overlay**: sample night prediction vs. expert.

## Interpretation & Progress
- **Met target**: Wake only.  
- **N1**: still far from ≥ 0.70 due to transient nature and imbalance.  
- **Other stages**: moderate gains (0.74–0.80 F1) but below 0.90 goal.  
- **Smoothing trade‑off**: improved stability, reduced N1 recall.

## Future Work
1. **Transition modeling**: HMM/CRF layers to enforce stage continuity.  
2. **Data augmentation**: generate synthetic N1 epochs (mixup, GANs).  
3. **Self‑supervised pretraining**: leverage unlabeled EEG for better embeddings.  
4. **Subject adaptation**: per‑subject fine‑tuning to capture individual patterns.  
5. **Multi‑modal fusion**: integrate EOG/EMG to disambiguate stages.

## Conclusion
Our CNN + Transformer pipeline with N1‑focused focal loss achieves 88 % overall accuracy yet only 0.34 F1 for N1 (target ≥ 0.70). While matching literature baselines in overall metrics, further work on temporal constraints and scarce‐class augmentation is essential to meet our stringent success criteria for clinical and at‑home sleep monitoring.
