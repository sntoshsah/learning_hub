# Orthogonality and Least Squares

## Introduction
Orthogonality plays a crucial role in understanding the geometry of vectors and solving optimization problems like least squares. Orthogonal vectors are perpendicular, and their dot product is zero. The least squares method minimizes the error in approximating a system of equations.

---

## Inner Product, Length, and Orthogonality

### Inner Product
The **inner product** (or dot product) of two vectors \( \mathbf{u}, \mathbf{v} \in \mathbb{R}^n \) is defined as:

\[
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
\]

### Properties
1. Commutative: \( \mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u} \)
2. Distributive: \( \mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w} \)
3. Scalar multiplication: \( (c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v}) \)

### Length
The length (or norm) of a vector \( \mathbf{v} \) is:

\[
\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}}
\]

### Orthogonality
Two vectors \( \mathbf{u} \) and \( \mathbf{v} \) are orthogonal if:

\[
\mathbf{u} \cdot \mathbf{v} = 0
\]

### Example
\( \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \mathbf{v} = \begin{bmatrix} -2 \\ 1 \end{bmatrix} \):

\[
\mathbf{u} \cdot \mathbf{v} = 1(-2) + 2(1) = 0 \quad \text{(orthogonal)}
\]

---

## Orthogonal Sets

### Definition
A set of vectors \( \{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n\} \) is **orthogonal** if:

\[
\mathbf{u}_i \cdot \mathbf{u}_j = 0 \quad \text{for all } i \neq j
\]

### Orthonormal Sets
An orthogonal set is **orthonormal** if each vector has unit length:

\[
\|\mathbf{u}_i\| = 1 \quad \text{for all } i
\]

### Example
\( \mathbf{u}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \mathbf{u}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \):

1. \( \mathbf{u}_1 \cdot \mathbf{u}_2 = 0 \quad \text{(orthogonal)} \)
2. \( \|\mathbf{u}_1\| = \|\mathbf{u}_2\| = 1 \quad \text{(orthonormal)} \).

---

## Orthogonal Projections

### Definition
The projection of \( \mathbf{y} \) onto \( \mathbf{u} \) is:

\[
\text{proj}_{\mathbf{u}} \mathbf{y} = \frac{\mathbf{y} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}
\]

### Example
\( \mathbf{y} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \):

\[
\text{proj}_{\mathbf{u}} \mathbf{y} = \frac{3(1) + 4(2)}{1^2 + 2^2} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \frac{11}{5} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2.2 \\ 4.4 \end{bmatrix}
\]

---

## The Gram-Schmidt Process

### Definition
The Gram-Schmidt process generates an orthogonal (or orthonormal) set of vectors from a linearly independent set \( \{\mathbf{v}_1, \mathbf{v}_2, \dots\} \).

### Steps
1. Set \( \mathbf{u}_1 = \mathbf{v}_1 \).
2. For \( k = 2, 3, \dots \):

    \[
    \mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \text{proj}_{\mathbf{u}_j}(\mathbf{v}_k)
    \]

3. Normalize \( \mathbf{u}_k \) to get an orthonormal set.

### Example
Given \( \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix} \):

1. \( \mathbf{u}_1 = \mathbf{v}_1 \).
2. \( \mathbf{u}_2 = \mathbf{v}_2 - \text{proj}_{\mathbf{u}_1} \mathbf{v}_2 \).

---

## Least Squares Problem

### Definition
The least squares solution minimizes the error \( \|A\mathbf{x} - \mathbf{b}\| \).

### Normal Equations
\[
A^T A \mathbf{x} = A^T \mathbf{b}
\]

### Example
For \( A = \begin{bmatrix} 1 & 1 \\ 1 & -1 \\ 1 & 2 \end{bmatrix}, \mathbf{b} = \begin{bmatrix} 2 \\ 0 \\ 5 \end{bmatrix} \):

1. Compute \( A^T A \) and \( A^T \mathbf{b} \).
2. Solve \( A^T A \mathbf{x} = A^T \mathbf{b} \).

---

# Applications of Linear Models

## Examples
1. **Data Fitting**: Using least squares to fit curves to data.
2. **Image Compression**: Leveraging orthogonal projections in PCA.
3. **Machine Learning**: Linear regression as a least squares problem.

---

## Inner Product Space

### Definition
An **inner product space** is a vector space with an inner product defined, satisfying:

1. Linearity in the first argument.
2. Symmetry: \( \langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle \).
3. Positive-definiteness: \( \langle \mathbf{u}, \mathbf{u} \rangle > 0 \) for \( \mathbf{u} \neq 0 \).

### Example
\( \mathbb{R}^n \) with \( \langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} \).

---

Feel free to expand sections with more examples or detailed applications!
