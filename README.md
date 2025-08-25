# OSI-and-Structured-Random-Matrices

This repository contains the `matlab`/`C` codebase for the paper **Faster Linear Algebra Algorithms with Structured Random Matrices** by _Chris Camaño, Ethan N. Epperly, Raphael A. Meyer, and Joel A. Tropp_. Avilable here [hrefarxiv link].  

<img width="5434" height="1534" alt="Histogram_final2" src="https://github.com/user-attachments/assets/8a0a4435-4584-4fc6-89be-9b0c0045eebd" />

---

Contents
---
### 1. Structured Sketching Matrices
---

- **SparseStacks**  
  Efficient `C` implementations of the *SparseStack* sketching matrix are provided in `code/sketches/SparseStack/sparseStack.c`. To compile locally in MATLAB, run:
  ```matlab
  mex sparseStack.c
  ```
  from the `code/sketches/SparseStack/` directory.  

- **Khatri–Rao**  
  Implementations of the Khatri–Rao sketching matrices, a tensor-product structured test matrix specified by a base distribution. Available options include real and complex Gaussians, real and complex Sphericals, real and complex Rademachers, and the Steinhaus distribution. Code is available at `code/sketches/KR/kr_sketch.m`.  

- **Subsampled Trigonometric Transforms**  
  Implementations of subsampled trigonometric transforms are available at `code/sketches/SparseTrigTransforms/srtt_explicit.c`. The current implementation explicitly forms the matrix
  $\Omega = \mathbf{D}\mathbf{F}\mathbf{S}$
  where **S** (the sparse component) can be chosen as uniform subsampling (default), SparseCol, CountSketch, or i.i.d. sampling. The transform **F** can be chosen as the Discrete Cosine Transform, Fast Fourier Transform, or Walsh–Hadamard Transform. A custom `C` implementation of the Walsh–Hadamard transform is provided at `code/sketches/SparseTrigTransforms/fwht.c` (compile with `mex fwht.c`).
### 2. Applications to Quantum Physics
---

<img width="1853" height="751" alt="GPE3" src="https://github.com/user-attachments/assets/aeabec77-8667-4133-8bd0-78ac065bb47e" />

- **Gross–Pitaevskii Data Compression**  
  [Simulation of the Gross–Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation), a nonlinear PDE governing the behavior of certain [Bose–Einstein condensates](https://en.wikipedia.org/wiki/Bose–Einstein_condensate). Proper Orthogonal Decomposition (POD) analysis and data compression are computed via the Generalized Nyström SVD with *SparseStack* test matrices. Experiment available at  
  `experiments/Applications/GPE/GPEsim.m`.

- **Partition Function Estimation with Variance-Reduced Stochastic Trace Estimation**  
  Estimation of the partition function  
  $Z(\beta) := \text{tr}\left(\exp(-\beta \mathbf{H})\right)$
  for the transverse-field Ising model Hamiltonian. Experiment available at  
  `experiments/Applications/Partition_function.m`.
  
<img width="3637" height="2988" alt="Sparse_final" src="https://github.com/user-attachments/assets/0e3dcd79-2cdc-43aa-bf09-213f461ef3a5" />

### 3. Low Rank Approximation experiments. 
Experiments testing the lowrank approximation ability of different structured test matrices can be found within `experiments/`. Plotting for all figures and additional content can be found in the `plotting/` directory which is done locally via jupyter notebooks.



