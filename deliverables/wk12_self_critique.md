04/18/2025

# Self Critique Week 12

## Progress Since Week 9  
- **Problem & Success Criteria Defined**  
  We’ve formalized our quantitative targets up front:  
  - **Primary**: N1 F1 > 0.70  
  - **Secondary**: F1 > 0.90 for Wake, N2, N3, REM; Overall accuracy > 88 %; Cohen’s κ > 0.80  
  We will consider this project a success if we meet or exceed these thresholds, tying our motivation (better N1 detection for clinical utility) directly to measurable outcomes.  

- **Revised Architecture**  
  - **Epoch Encoder**: 3 × 1D‑CNN layers → 128‑d embedding  
  - **Positional Encoding**: learned 20 × 128 tensor  
  - **Transformer Encoder**: 2 layers, 4 heads, 512‑dim feed‑forward  
  - **Classification Head**: linear map → 5 classes + median‑filter smoothing (kernel = 5)  
  - **Loss**: focal loss (αₙ₁ = 0.75, αₒₜₕₑᵣ = 0.25; γ = 2) to aggressively target N1 misclassification  

## Achievements  
- **Pipeline Implementation**  
  - End‑to‑end data processing (MNE for filtering, epoching; sliding windows of 20 × 30 s epochs)  
  - True subject‑wise train/val/test splits to prevent leakage  
  - Modular codebase ready for cloud scaling on GCP  

- **Baseline Performance**  
  - Overall accuracy = 88 %  
  - Wake, N2, N3, REM F1 ≥ 0.80  
  - **N1 F1 ≈ 0.39** (pre‑smoothing), 0.34 (post‑smoothing)  

## Remaining Challenges  
- **N1 Detection Gap**  
  Despite strong overall staging, N1 F1 remains far below our 0.70 target.  
- **Class Imbalance & Ambiguity**  
  N1 epochs still < 5 % of data; spectral overlap with Wake/N2 persists.  
- **Compute & Scalability**  
  Local M1 MacBook limits batch size and hyperparameter sweeps; moving to GCP is essential.  

## Next Steps  
1. **Incorporate Domain Constraints**  
   - Learnable state‑space or HMM layer on top of Transformer to enforce physiologically plausible transitions.  
2. **Enhanced Context Modeling**  
   - Experiment with longer sequences (30–40 epochs) or hierarchical Transformers to capture broader sleep architecture.  
3. **Targeted Augmentation**  
   - Apply temporal jittering and spectral‑preserving noise to synthesize more diverse N1 examples.  
4. **Subject Adaptation**  
   - Implement domain‑adversarial training or per‑subject fine‑tuning to handle inter‑subject variability.  
5. **Cloud Scaling & Validation**  
   - Containerize pipeline for GCP; run cross‑dataset tests on MASS and SHHS for proof of concept.  

## Team Workflow Improvements  
- Established weekly “sync‑digest” docs to reduce asynchronous friction and ensure all members are up‑to‑date.  
- Defined clear module ownership (data, model, evaluation, infra) and pull‑request templates to streamline contributions.  

## Ensemble Model Focus  
- **Previous Supervised‑Feature Successes**  
  - On the ANPHY dataset, classical classifiers (Logistic Regression, SVM, Random Forest, XGBoost) using hand‑crafted features (spectral power, Catch22 set) achieved F1 > 0.85 for Wake/N2/N3/REM but N1 F1 remained ≈ 0.06.  
  - Our deep‑learning model improved N1 F1 to ≈ 0.39, but still below target.  

- **Bottleneck**  
  - We’ve yet to combine these two complementary approaches—hand‑crafted domain features capture slow‑wave/spindle events that the CNN/Transformer can miss.  

- **Goal**  
  - Build a stacked or feature‑fusion ensemble that merges Catch22/statistical features with our Transformer embeddings.  
  - Tune ensemble weights or meta‑learner to boost N1 detection without sacrificing overall staging.  

## Final Stretch Objectives  
- **Implement & Evaluate Ensemble**  
  - Prototype a meta‑classifier (e.g., gradient‑boosted tree) on concatenated feature+embedding vectors.  
  - Compare ensemble N1 F1 against individual models on held‑out Sleep‑EDF test set.  
- **Hyperparameter Search**  
  - Grid/search ensemble mixing coefficients and meta‑learner depth.  
- **Visualization & Explainability**  
  - Use SHAP or similar to interpret which features (hand‑crafted vs. learned) drive N1 predictions.  

By targeting this ensemble integration in the final weeks, we aim to get our N1 F1 beyond 0.70 while preserving ≥ 0.90 F1 on the other stages.  