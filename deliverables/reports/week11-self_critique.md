This week, we made several experimental attempts to improve our sleep stage classification model, focusing primarily on enhancing N1 detection. Our initial approach involved modifying the CNN encoder by integrating multi-scale convolutional kernels and incorporating class weighting to address class imbalance. Although conceptually promising, these modifications did not yield the expected improvements; in fact, overall performance suffered, suggesting that the increased architectural complexity might have hindered effective feature learning.

In response, we pivoted to alternative strategies. We explored both non-targeted and targeted one‑vs‑all classifier implementations. The targeted approach—focusing exclusively on N1—integrated a dedicated binary classifier with a residual boosting head to correct the N1 logit. This refinement showed potential in improving N1 discrimination. Additionally, we enriched our model by incorporating engineered features, such as catch22 and power/spectral density metrics, which were fused with the transformer's learned features through a gating mechanism. This allowed the model to dynamically balance between learned and engineered features.

However, while these enhancements offered valuable insights, they also introduced significant computational overhead. Running these experiments on Google Colab led to frequent runtime disconnects, highlighting the need for more efficient feature extraction—perhaps by precomputing the engineered features or optimizing the extraction process.

Overall, we saw a bit of improvement but there is still much we should be doing, such as: 

Incorporate (more) Domain Knowledge: Integrate control‑theoretic or dynamical systems constraints (e.g., a learned transition matrix or a state‑space model) to enforce physiologically plausible stage transitions. This could help distinguish the subtle differences at the wake–N1 boundary.

Enhanced Context Modeling: Further refine our sequence‑to‑sequence Transformer architecture to better capture long‑range dependencies. For example, experimenting with longer sequences or using hierarchical models might help provide additional context that clarifies ambiguous N1 epochs.

Fine‑Tuning on Fragmented Sleep Data: Since N1 detection is especially challenging in subjects with fragmented sleep, developing a sub‑model or fine‑tuning strategy that targets these cases specifically might yield improvements.

Advanced Data Augmentation: While synthetic data is limited in our case, exploring advanced augmentation techniques (e.g., temporal jittering, noise injection that preserves key spectral features) may increase the diversity of N1 examples and improve model robustness. This could also help with the data imbalance issue we were having in another way other than stratified sampling.

Inter-Subject Variability Handling: Implementing subject adaptation or domain adversarial training to better capture the individual variability in EEG patterns could help the model generalize its detection of N1 onset.
