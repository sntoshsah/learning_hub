---
title: Vector Spaces
description: A detailed exploration of vector spaces, their properties, and applications in linear algebra.
---

# Vector Spaces

## 1. Vector Space and Subspace

### Introduction
A **vector space** is a set of vectors that can be added together and multiplied by scalars, satisfying specific axioms (closure, associativity, distributivity, etc.). Vector spaces provide the framework for linear algebra and are crucial for applications in mathematics, physics, and engineering.

A **subspace** is a subset of a vector space that is itself a vector space under the same operations.

### Key Concepts

- **Vector Space**: A set \( V \) with operations addition and scalar multiplication such that:

  1. Closure under addition and scalar multiplication.
  2. Existence of a zero vector.
  3. Additive inverses exist.
  4. Associativity and commutativity of addition.
  5. Distributive properties.

- **Subspace Criteria**:

  1. The zero vector is in the subset.
  2. Closed under vector addition.
  3. Closed under scalar multiplication.

### Example
1. The set of all vectors \( \mathbb{R}^n \) is a vector space.
2. The set of all solutions to a homogeneous linear equation \( Ax = 0 \) forms a subspace.

### Visualization
Subspaces of \( \mathbb{R}^3 \):
- A line through the origin.
- A plane through the origin.

---

## 2. Null Space and Column Space, and Linear Transformation

### Null Space
The **null space** of a matrix \( A \) is the set of all vectors \( \mathbf{x} \) such that:

\[
A\mathbf{x} = \mathbf{0}
\]

### Column Space
The **column space** is the span of the columns of \( A \). It represents all possible linear combinations of the columns.

### Linear Transformation
A **linear transformation** is a mapping \( T: V \to W \) such that:

1. \( T(u + v) = T(u) + T(v) \)
2. \( T(cu) = cT(u) \)

### Example
Given \( A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \):

- Null space: Solve \( A\mathbf{x} = \mathbf{0} \).
- Column space: Span of \( \begin{bmatrix} 1 \\ 3 \\ 5 \end{bmatrix}, \begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix} \).

### Code Example
```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2], [3, 4], [5, 6]])
null = null_space(A)
print("Null Space:", null)
```

---

## 3. Linearly Independent Sets and Bases

### Linearly Independent Sets
A set of vectors \( \{v_1, v_2, \dots, v_k\} \) is linearly independent if:

\[
c_1v_1 + c_2v_2 + \dots + c_kv_k = 0 \implies c_1 = c_2 = \dots = c_k = 0
\]

### Bases
A **basis** is a linearly independent set of vectors that spans the vector space.

### Dimension
The number of vectors in a basis is the **dimension** of the vector space.

### Example
The standard basis for \( \mathbb{R}^3 \):

\[
\{ \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \}
\]

### Code Example
```python
A = np.array([[1, 2], [3, 4]])
rank = np.linalg.matrix_rank(A)
print("Rank (dimension of column space):", rank)
```

---

## 4. Coordinate System

### Definition
A **coordinate system** represents a vector as a linear combination of basis vectors.

### Change of Basis
To convert coordinates from one basis \( B \) to another \( C \):

\[
[\mathbf{x}]_C = P_{B \to C} [\mathbf{x}]_B
\]

Where \( P_{B \to C} \) is the change of basis matrix.

### Example
Convert \( \mathbf{x} \) from basis \( B = \{\mathbf{b}_1, \mathbf{b}_2\} \) to \( C = \{\mathbf{c}_1, \mathbf{c}_2\} \).

### Code Example
```python
B = np.array([[1, 0], [0, 1]])
C = np.array([[2, 1], [1, 3]])
P = np.linalg.inv(B) @ C
print("Change of Basis Matrix:", P)
```

---

## Conclusion
Vector spaces form the backbone of linear algebra, connecting concepts like independence, transformations, and applications in numerous fields. Understanding these fundamentals is crucial for advanced studies and practical problem-solving.

---

## Exercises

1. Prove that the set of all polynomials of degree at most 2 forms a vector space.
2. Find the null space and column space of the matrix:

    \[
    A = \begin{bmatrix} 2 & 4 \\ -1 & -2 \\ 3 & 6 \end{bmatrix}
    \]

3. Verify if the vectors \( \{(1, 0, 0), (0, 1, 0), (1, 1, 1)\} \) are linearly independent.
4. Compute the coordinates of \( \mathbf{x} = \begin{bmatrix} 3 \\ 7 \end{bmatrix} \) relative to the basis \( \{\begin{bmatrix} 1 \\ 2 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}\} \).

Solutions available upon request!
