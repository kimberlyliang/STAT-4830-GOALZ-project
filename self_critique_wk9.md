03/21/2025

I believe we have made reasonable progress with respect to narrowing down the particular problem we are trying to solve. We have conducted an extensive review of the literature, identifying the architectures that best address class imbalance and specifically the challenge of identifying N1 onset. Our experimental pipeline now integrates state‑of‑the‑art preprocessing, including splitting the Sleep‑EDF data into epochs and sequences, and a novel end‑to‑end model that combines a CNN encoder with a Transformer for sequence modeling. While it is encouraging that our models can match and sometimes outperform SOTA in overall sleep staging accuracy, we have yet to make substantial improvement on N1 onset detection in particular.

Since synthetic data does not seem to be an option in our case, it seems like we are left with the following options to improve N1 detection further:

Incorporate Domain Knowledge:
Integrate control‑theoretic or dynamical systems constraints (e.g., a learned transition matrix or a state‑space model) to enforce physiologically plausible stage transitions. This could help distinguish the subtle differences at the wake–N1 boundary.

Enhanced Context Modeling:
Further refine our sequence‑to‑sequence Transformer architecture to better capture long‑range dependencies. For example, experimenting with longer sequences or using hierarchical models might help provide additional context that clarifies ambiguous N1 epochs.

Fine‑Tuning on Fragmented Sleep Data:
Since N1 detection is especially challenging in subjects with fragmented sleep, developing a sub‑model or fine‑tuning strategy that targets these cases specifically might yield improvements.

Advanced Data Augmentation:
While synthetic data is limited in our case, exploring advanced augmentation techniques (e.g., temporal jittering, noise injection that preserves key spectral features) may increase the diversity of N1 examples and improve model robustness.

Inter-Subject Variability Handling:
Implementing subject adaptation or domain adversarial training to better capture the individual variability in EEG patterns could help the model generalize its detection of N1 onset.

Moreover, we have struggled to coordinate and determine and efficient mechanism to distribute work when we are working asynchronously, introducing substantial friction into our workflow and ultimately slowing down our productivity (don't tell DOGE). Hence, moving forward, we will need to develop a more streamlined way of passing updates about what we have been up to most recently to our other teammates so that others don't need to waste time 'catching up' on what has been done.

Additionally, now that we have been granted google cloud access, we should make sure to develop our pipeline more modularly so that we will be able to run additional analyses on validation datasets, which is crucial for proof-of-concept of our implementation.