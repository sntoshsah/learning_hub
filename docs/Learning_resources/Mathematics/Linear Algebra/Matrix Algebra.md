---
title: Matrix Algebra
description: A comprehensive guide to understanding matrix algebra with detailed explanations, formulas, examples, and applications.
---

# Matrix Algebra

## 1. Introduction
**Matrix Algebra** is the branch of mathematics dealing with operations on matrices. Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are widely used in various fields, including engineering, physics, economics, and computer graphics.

---

## 2. Matrix Operations
Matrix operations include addition, subtraction, scalar multiplication, and matrix multiplication.

### Addition and Subtraction
For matrices \(A = [a_{ij}]\) and \(B = [b_{ij}]\) of the same size:

\[
C = A + B \quad \text{where} \quad c_{ij} = a_{ij} + b_{ij}
\]

\[
D = A - B \quad \text{where} \quad d_{ij} = a_{ij} - b_{ij}
\]

### Scalar Multiplication
For a scalar \(k\):

\[
kA = [ka_{ij}]
\]

### Matrix Multiplication
For matrices \(A (m \times n)\) and \(B (n \times p)\):

\[
C = AB \quad \text{where} \quad c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}
\]

### Code Example
```python
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B

# Matrix multiplication
D = A @ B

print("Addition:\n", C)
print("Multiplication:\n", D)
```

---

## 3. The Inverse of the Matrix
The inverse of a square matrix \(A\) is denoted \(A^{-1}\), where:

\[
A A^{-1} = A^{-1} A = I
\]

\(I\) is the identity matrix.

### Formula for 2x2 Matrix
For \(A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\):

\[
A^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}, \quad \text{if } ad - bc \neq 0
\]

### Code Example
```python
# Define a matrix
A = np.array([[4, 7], [2, 6]])

# Compute the inverse
A_inv = np.linalg.inv(A)
print("Inverse:\n", A_inv)
```

---

## 4. Characterizations of Invertible Matrices
A square matrix \(A\) is invertible if:

1. \(\det(A) \neq 0\).
2. \(A\) has full rank.
3. The rows or columns of \(A\) are linearly independent.

---

## 5. Partitioned Matrices
Partitioned matrices divide a matrix into smaller submatrices for easier computations.

### Example
For \(A\):

\[
A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
\]

This structure is useful for block multiplication and solving systems of equations.

---

## 6. Matrix Factorizations
Matrix factorizations decompose matrices into simpler forms:

### LU Factorization
\(A = LU\), where:

- \(L\): Lower triangular matrix.
- \(U\): Upper triangular matrix.

### QR Factorization
\(A = QR\), where:

- \(Q\): Orthogonal matrix.
- \(R\): Upper triangular matrix.

### Code Example
```python
from scipy.linalg import lu

# Define a matrix
A = np.array([[4, 3], [6, 3]])

# LU decomposition
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)
```

---

## 7. The Leontief Input/Output Model
This economic model represents interdependencies between sectors:

\[
x = (I - A)^{-1} d
\]

Where:

- \(A\): Input-output matrix.
- \(d\): Final demand vector.

---

## 8. Applications of Matrix Algebra to Computer Graphics
Matrices are essential in transforming coordinates in computer graphics:

### Translation

\[
T = \begin{bmatrix} 1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1 \end{bmatrix}
\]

### Rotation

\[
R = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1 \end{bmatrix}
\]

### Code Example
```python
# Define a 2D rotation matrix
import math

angle = math.pi / 4  # 45 degrees
R = np.array([[math.cos(angle), -math.sin(angle)],
              [math.sin(angle), math.cos(angle)]])

# Rotate a point
point = np.array([1, 0])
rotated_point = R @ point
print("Rotated Point:", rotated_point)
```

---

## 9. Subspaces of \(\mathbb{R}^n\)
Subspaces are subsets of \(\mathbb{R}^n\) closed under addition and scalar multiplication.

### Examples
- Column space: Set of all linear combinations of columns.
- Null space: Set of all solutions to \(A\mathbf{x} = 0\).

---

## 10. Dimension and Rank
The **dimension** of a subspace is the number of vectors in its basis. The **rank** of a matrix is the dimension of its column space.

### Rank-Nullity Theorem

\[
\text{rank}(A) + \text{nullity}(A) = \text{number of columns of } A
\]

### Code Example
```python
# Define a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute rank
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)
```

---

This guide covers fundamental concepts, practical examples, and Python implementations to help you master matrix algebra.
