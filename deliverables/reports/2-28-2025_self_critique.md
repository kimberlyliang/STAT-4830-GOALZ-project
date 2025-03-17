**Self-Critique**
*Data Sampling Limitations*
Restricted Epoch Selection: Currently, we pick only one Wake epoch and one N1 epoch per subject for the binary classification task. This limited sampling strategy fails to capture the full range of variability within and across subjects. We risk training models that may not generalize well when faced with more complex or nuanced data (e.g., multiple transitions per night, variation in sleep architecture).
Potential Class Imbalance: Although selecting one W and one N1 epoch per subject yields a balanced dataset for two classes, it neglects real‐world scenarios where the number of Wake vs. N1 epochs can vary widely across a full night’s recording.

*Lack of a Concrete Mathematical Formulation*
Vague Optimization Objective: While we aim to maximize classification performance (accuracy, AUC) and minimize misclassifications of Wake vs. N1, we have not yet formalized a clear objective function. A more rigorous mathematical statement is needed to ground the project in a reproducible, theoretically sound framework.

*Window Size and Temporal Context*
2‐Second Snippets: By using very short 2‐second windows (±1 second) to represent each labeled epoch, we may be discarding critical context needed to detect subtle transitions or micro‐arousals. Longer windows (e.g., 5 or 10 seconds) or overlapping windows could potentially enhance classification performance, but we have not systematically tested these variations.

*Downsampling and Frequency Content*
Optimal Sampling Rate: We downsample from 1000 Hz to 200 Hz to reduce data size, but we have not conclusively determined whether 200 Hz best preserves essential sleep onset information. Downsampling too aggressively may lose critical frequency components, while insufficient downsampling can cause inefficiencies in processing and training.
Filter Design: We currently apply a 60 Hz notch filter and a 4th‐order lowpass filter at 90 Hz. The extent to which these filter choices alter the EEG signal—particularly around known sleep spindles or other micro‐patterns—remains an open question.

*Scalability to Full Nights and Multi‐Stage Classification*
Memory Constraints: Our pipeline handles a small subset of labeled epochs effectively, but scaling up to entire nights (with thousands of 30‐second epochs) or more subjecets (e.g., MESA) could overwhelm current memory handling strategies. We need more efficient chunked loading and streaming techniques.
Multi‐Stage Complexity: Transitioning beyond W vs. N1 to a full 5‐class classification (W, N1, N2, N3, REM) introduces additional label imbalance, stage transition nuances, and higher model complexity. The performance gains we see in binary classification might not straightforwardly translate to a multi‐stage task without significant pipeline modifications.

*Generalizability and External Validation*
Limited Testing Across Subjects: We have not yet validated how well our models—trained on a handful of epochs from each subject—scale to unseen participants or different recording conditions. Overfitting to a small set of epochs is a real possibility.
Cross‐Dataset Applicability: Although we intend to use MESA (a larger public dataset) for validation, the current approach has only been tested on the smaller ANPHY dataset. Transferability and reproducibility in other EEG datasets is still uncertain.

*Potential Overfitting and Model Selection*
High Dimensionality vs. Few Samples: Even with classical models like logistic regression, SVM, and random forest, high‐dimensional EEG data and a small number of labeled epochs invite a higher risk of overfitting. We must refine our sampling and cross‐validation protocols to ensure the model learns features that generalize.
Future Use of Deep Learning: We have postponed exploring advanced neural architectures due to computational constraints, but now we are more ready to move to them once we have a robust pipeline for full nights of data.
