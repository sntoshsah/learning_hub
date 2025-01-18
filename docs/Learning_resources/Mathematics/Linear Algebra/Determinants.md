---
title: Determinants
description: A comprehensive guide to understanding determinants, their properties, and applications such as Cramer's Rule, volume calculations, and linear transformations.
---

# Determinants

## 1. Introduction to Determinants
A **determinant** is a scalar value associated with a square matrix. It provides important information about the matrix, such as invertibility, and plays a crucial role in linear algebra.

### Notation
For a square matrix \(A\):

\[
\det(A) \quad \text{or} \quad |A|
\]

### Determinants of Small Matrices
#### 2x2 Matrix
For \(A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\):

\[
\det(A) = ad - bc
\]

#### 3x3 Matrix
For \(A = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}\):

\[
\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
\]

### Applications of Determinants
1. Determining invertibility (\(\det(A) \neq 0\) implies \(A\) is invertible).
2. Calculating volumes of parallelepipeds.
3. Solving systems of linear equations using Cramer’s Rule.
4. Analyzing the effects of linear transformations.

---

## 2. Properties of Determinants
Determinants have several useful properties that simplify calculations and provide insight into matrix behavior.

### Key Properties
1. **Determinant of Identity Matrix**:

    \[
    \det(I) = 1
    \]

2. **Row or Column Swapping**:
   Swapping two rows (or columns) changes the sign of the determinant.

3. **Scalar Multiplication**:
   If a row (or column) is multiplied by a scalar \(k\):

   	\[
   	\det(A') = k \det(A)
   	\]

4. **Additive Property**:
   Adding a multiple of one row to another does not change the determinant.

5. **Zero Row or Column**:
   If a matrix has a row or column of all zeros:

    \[
    \det(A) = 0
    \]

6. **Upper or Lower Triangular Matrix**:
   For triangular matrices, the determinant is the product of the diagonal elements:

   	\[
   	\det(A) = a_{11}a_{22}\cdots a_{nn}
   	\]

### Code Example
```python
import numpy as np

# Define a matrix
A = np.array([[2, 1], [5, 3]])

# Compute the determinant
det_A = np.linalg.det(A)
print("Determinant:", round(det_A))
```

---

## 3. Cramer’s Rule, Volume, and Linear Transformations

### Cramer’s Rule
Cramer’s Rule solves systems of linear equations \(A\mathbf{x} = \mathbf{b}\) using determinants.

#### Formula
For \(n \times n\) matrix \(A\):\

\[
x_i = \frac{\det(A_i)}{\det(A)}
\]

Where \(A_i\) is the matrix formed by replacing the \(i\)-th column of \(A\) with \(\mathbf{b}\).

#### Example
Solve:

\[
\begin{cases}
2x + 3y = 8 \\
4x + y = 10
\end{cases}
\]

**Solution:**

\[
A = \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 8 \\ 10 \end{bmatrix}
\]

\[
\det(A) = (2)(1) - (3)(4) = -10
\]

\[
A_x = \begin{bmatrix} 8 & 3 \\ 10 & 1 \end{bmatrix}, \quad A_y = \begin{bmatrix} 2 & 8 \\ 4 & 10 \end{bmatrix}
\]

\[
x = \frac{\det(A_x)}{\det(A)} = \frac{-14}{-10} = 1.4, \quad y = \frac{\det(A_y)}{\det(A)} = \frac{-20}{-10} = 2
\]

### Code Example
```python
# Define matrices
A = np.array([[2, 3], [4, 1]])
b = np.array([8, 10])

# Solve using Cramer's Rule
det_A = np.linalg.det(A)
x1 = np.linalg.det(np.column_stack((b, A[:, 1]))) / det_A
x2 = np.linalg.det(np.column_stack((A[:, 0], b))) / det_A

print("Solution:", (x1, x2))
```

---

### Volume and Determinants
The determinant of a matrix formed by vectors \(\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\) represents the volume of the parallelepiped they span.

#### Formula
For vectors \(\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\):

\[
\text{Volume} = |\det([\mathbf{v}_1 \ \mathbf{v}_2 \ \cdots \ \mathbf{v}_n])|
\]

#### Example
Calculate the volume of a parallelepiped spanned by:

\[
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 0 \\ 0 \\ 2 \end{bmatrix}
\]

**Solution:**

\[
\text{Matrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 2 \end{bmatrix}
\]

\[
\text{Volume} = |\det(A)| = |1 \cdot 1 \cdot 2| = 2
\]

### Code Example
```python
# Define vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 2])

# Form matrix and compute volume
A = np.column_stack((v1, v2, v3))
volume = abs(np.linalg.det(A))
print("Volume:", volume)
```

---

### Determinants in Linear Transformations
The determinant of a transformation matrix \(T\) indicates how the transformation scales areas or volumes:

- \(\det(T) > 0\): Preserves orientation.
- \(\det(T) < 0\): Reverses orientation.
- \(\det(T) = 0\): Flattens the space (non-invertible).

---

This guide covers determinants in-depth with examples, properties, and practical applications. Let me know if you need additional sections or clarifications!
