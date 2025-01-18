# Introduction to Eigenvalues and Eigenvectors

## What Are Eigenvalues and Eigenvectors?

**Eigenvalues** and **eigenvectors** are fundamental concepts in linear algebra, appearing in many areas such as physics, computer science, and engineering.

### Definition
- **Eigenvector**: A non-zero vector \( \mathbf{v} \) such that when a linear transformation (matrix \( A \)) is applied to it, the vector is scaled by a scalar \( \lambda \):

	\[ A \mathbf{v} = \lambda \mathbf{v} \]

- **Eigenvalue**: The scalar \( \lambda \) that corresponds to an eigenvector \( \mathbf{v} \).

### Key Properties
1. Eigenvectors corresponding to distinct eigenvalues are linearly independent.
2. If \( A \) is an \( n \times n \) matrix, there are at most \( n \) eigenvalues (some may be repeated).

### Example
Given:

\[
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\]

Find eigenvalues and eigenvectors.

### Solution

1. Compute \( \det(A - \lambda I) \):

    \[
    \det\left(\begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}\right) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3
    \]
 
    Solve \( \lambda^2 - 4\lambda + 3 = 0 \) to get \( \lambda_1 = 3 \), \( \lambda_2 = 1 \)
    
2. Find eigenvectors for each eigenvalue:
    - For \( \lambda = 3 \): Solve \( (A - 3I) \mathbf{v} = 0 \).
    - For \( \lambda = 1 \): Solve \( (A - I) \mathbf{v} = 0 \).

---

## The Characteristic Equation
### Definition
The **characteristic equation** of a square matrix \( A \) is derived from \( \det(A - \lambda I) = 0 \). This polynomial equation determines the eigenvalues of \( A \).

### Steps to Formulate
1. Subtract \( \lambda I \) from \( A \):

	\[ A - \lambda I \]

2. Compute the determinant:

	\[ \det(A - \lambda I) \]

3. Set \( \det(A - \lambda I) = 0 \).

### Example
For \( A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix} \):

1. \( A - \lambda I = \begin{bmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{bmatrix} \).
2. \( \det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 \cdot 1 = \lambda^2 - 7\lambda + 10 \).
3. Solve \( \lambda^2 - 7\lambda + 10 = 0 \) to get \( \lambda = 5, 2 \).

---

## Diagonalization

### What Is Diagonalization?
A matrix \( A \) is **diagonalizable** if it can be expressed as:

\[ A = PDP^{-1} \]

where:

- \( P \) is a matrix whose columns are the eigenvectors of \( A \).
- \( D \) is a diagonal matrix with eigenvalues of \( A \) on its diagonal.

### Conditions for Diagonalization
1. \( A \) must have \( n \) linearly independent eigenvectors.
2. \( A \) must be a square matrix.

### Steps to Diagonalize
1. Find eigenvalues \( \lambda_1, \lambda_2, \dots \).
2. Find eigenvectors for each eigenvalue.
3. Form \( P \) using eigenvectors as columns.
4. Form \( D \) with eigenvalues along the diagonal.
5. Compute \( P^{-1} \).

### Example
For \( A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \):

1. Eigenvalues: \( \lambda_1 = 5, \lambda_2 = 2 \).
2. Eigenvectors: \( \mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix} \).
3. \( P = \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix} \), \( D = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix} \).
4. \( P^{-1} = \begin{bmatrix} 1/3 & 1/3 \\ -2/3 & 1/3 \end{bmatrix} \).
5. Verify: \( A = PDP^{-1} \).

---

## Eigenvectors and Linear Transformations

### Geometric Interpretation
An eigenvector represents a direction that remains unchanged under the linear transformation defined by \( A \), except for scaling by the eigenvalue \( \lambda \).

### Applications
- **Principal Component Analysis (PCA)**: Eigenvectors represent principal directions of data variance.
- **Quantum Mechanics**: Eigenvalues correspond to measurable quantities.
- **Graph Theory**: Eigenvectors indicate centrality in networks.

### Example
For a transformation \( A \):

\[ A \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 2x + y \\ x + 2y \end{bmatrix} \]

Find eigenvalues and interpret geometrically.

---

# Complex Eigenvalues

## Definition
If \( A \) has complex eigenvalues, they appear in conjugate pairs \( \lambda = a \pm bi \).

### Example
For \( A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \):
1. \( \det(A - \lambda I) = \lambda^2 + 1 = 0 \).
2. Eigenvalues: \( \lambda = i, -i \).
3. Eigenvectors involve complex numbers, e.g., \( \mathbf{v} = \begin{bmatrix} i \\ 1 \end{bmatrix} \).

---

## Discrete Dynamical Systems

### Definition
A **discrete dynamical system** evolves in discrete time steps according to:

\[ \mathbf{x}_{n+1} = A \mathbf{x}_n \]

where \( A \) is the transition matrix.

### Stability Analysis
1. Compute eigenvalues of \( A \).
2. If all \( |\lambda| < 1 \), the system is stable.

### Example
For \( A = \begin{bmatrix} 0.5 & 0.5 \\ 0.2 & 0.8 \end{bmatrix} \):
1. Eigenvalues: \( \lambda = 1, 0.3 \).
2. System is stable as \( |\lambda| < 1 \).

### Applications
- Population models
- Economic systems
- Markov chains
