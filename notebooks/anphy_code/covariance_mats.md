Math and Rationale
(BE 6100 paper)
 • Every EEG window yields a data matrix X (with 93 channels and T time samples) from which you compute the sample covariance
   C = (1/(T–1)) · X · Xᵀ
  This C is a symmetric positive‐definite (SPD) matrix. Because the set of all symmetric matrices is a finite‑dimensional vector space, the subset of SPD matrices is an open set—and hence a differentiable manifold.
 • On this manifold, a natural (affine‑invariant) Riemannian metric is defined. For two SPD matrices C₁ and C₂, the Riemannian distance is given by
   δ_R(C₁, C₂) = ‖logm(C₁^–½ · C₂ · C₁^–½)‖_F
  where logm(·) denotes the matrix logarithm and ‖·‖F is the Frobenius norm.
 • To “linearize” the manifold locally, one projects an SPD matrix C onto the tangent space at a reference point C_ref via the logarithmic map:
   f(C) = Log{C_ref}(C) = C_ref^–½ · logm(C_ref^–½ · C · C_ref^–½) · C_ref^–½
  In this Euclidean tangent space, you can compute inner products (e.g., via the Frobenius product), leading to a kernel of the form
   k_R(C_i, C_j; C_ref) = ⟨f(C_i), f(C_j)⟩ = tr( f(C_i)ᵀ f(C_j) )
 • This approach respects the curved geometry of SPD matrices so that “distances” and similarities capture the true relationships between the covariance features rather than forcing them into a flat, Euclidean space.

Incorporating This into Your Pipeline
 • Since your sleep EEG recordings are long (∼8 hours at 1000 Hz, with potential downsampling), you’d first segment the recordings into shorter, stationary windows (e.g., a few seconds or epochs).
 • For each window, compute the 93×93 covariance matrix. This matrix captures the inter-channel (electrode) relationships over that segment.
 • Because you’ve observed that amplitude ranges and other “artificial” differences vary across subjects, mapping each covariance matrix to a common tangent space (using a reference such as the geometric mean of your training set or the identity matrix after whitening) can help remove these systematic biases.
 • You can then either use the tangent space representations directly as features for a classifier (like an SVM) or compute the Riemannian kernel between matrices. This step “normalizes” the data by taking into account the manifold’s curvature, which is especially useful when the transition between wakefulness and early sleep (N1) is subtle and hard to distinguish in the raw signal.
 • The computational cost is reasonable: computing covariance matrices and the subsequent eigenvalue decompositions or matrix logarithms on 93×93 matrices is quite feasible

Another paper:
 • The paper by Barachant et al. proposes a new kernel for classifying EEG trials based directly on their spatial covariance matrices.
 • Because each trial is represented by a covariance matrix—which is symmetric and (if estimated robustly) positive definite—the authors work on the space P(E) of SPD matrices.
 • This space is not flat; it has a curved (Riemannian) geometry. By leveraging an affine‑invariant Riemannian metric, the method computes distances via
  δ_R(C₁, C₂) = ‖logm(C₁^–½ C₂ C₁^–½)‖_F,
  which respects the intrinsic geometry of the data.
 • Mapping each SPD matrix onto the tangent space (using the logarithmic map at a reference point) yields a Euclidean representation where an inner product (and thus a kernel) can be defined.
 • Using this Riemannian-based kernel within an SVM framework, the method outperforms classical approaches (like CSP with LDA) for motor imagery classification in BCIs.