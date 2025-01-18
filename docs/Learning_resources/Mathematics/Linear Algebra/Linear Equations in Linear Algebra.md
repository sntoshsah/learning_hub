---
title: Linear Equations in Linear Algebra
description: A detailed guide to understanding Linear Equations in Linear Algebra, with explanations, mathematical examples, and tips for problem-solving.
---

# Linear Equations in Linear Algebra

## 1. System of Linear Equations
A **system of linear equations** consists of two or more linear equations involving the same set of variables. A solution to the system is a set of values for the variables that satisfy all equations simultaneously.

### General Form
A system of \(m\) linear equations in \(n\) variables can be written as:

\[
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
\]

where:

- \(a_{ij}\) are the coefficients of the variables \(x_1, x_2, \ldots, x_n\).
- \(b_i\) are the constants.

### Example
Solve the system of equations:

\[
\begin{aligned}
2x + 3y &= 8 \\
4x - y &= 2
\end{aligned}
\]

### Solution
1. Multiply the second equation by 3 to eliminate \(y\):

\[
12x - 3y = 6
\]

2. Add to the first equation:

\[
(2x + 3y) + (12x - 3y) = 8 + 6 \\
14x = 14 \\
x = 1
\]

3. Substitute \(x = 1\) into the first equation:

\[
2(1) + 3y = 8 \\
3y = 6 \\
y = 2
\]

Thus, the solution is \(x = 1, y = 2\).

---

## 2. Row Reduction and Echelon Forms
**Row reduction** transforms a matrix into a simpler form to solve linear systems. The two key forms are:

### Row Echelon Form (REF)
1. Each leading entry in a row is to the right of the leading entry in the row above.
2. All entries below a leading entry are zero.

### Reduced Row Echelon Form (RREF)
1. The matrix is in REF.
2. Each leading entry is 1.
3. Each leading 1 is the only nonzero entry in its column.

### Steps for Gaussian Elimination
1. Write the augmented matrix of the system.
2. Use row operations to achieve REF.
3. (Optional) Continue to RREF for simplicity.

#### Example
Solve the system:

\[
\begin{aligned}
x + y + z &= 6 \\
2y + 5z &= -4 \\
2x + 5y - z &= 27
\end{aligned}
\]

**Augmented Matrix:**

\[
\begin{bmatrix}
1 & 1 & 1 & | & 6 \\
0 & 2 & 5 & | & -4 \\
2 & 5 & -1 & | & 27
\end{bmatrix}
\]

Perform row reduction to solve:

\[
\begin{bmatrix}
1 & 1 & 1 & | & 6 \\
0 & 2 & 5 & | & -4 \\
0 & 3 & -3 & | & 15
\end{bmatrix}
\rightarrow 
\begin{bmatrix}
1 & 1 & 1 & | & 6 \\
0 & 1 & 5/2 & | & -2 \\
0 & 0 & -15/2 & | & 21
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 0 & 0 & | & 1 \\
0 & 1 & 0 & | & 2 \\
0 & 0 & 1 & | & -1
\end{bmatrix}
\]

Solution: \(x = 1, y = 2, z = -1\).

---

## 3. Vector Equations
A vector equation is an equation involving vectors and their linear combinations.

### General Form

\[
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{b}
\]

### Example
Given:

\[
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}, \mathbf{b} = \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}
\]

Find scalars \(c_1\) and \(c_2\) such that:

\[
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{b}
\]

Write as a system:

\[
\begin{aligned}
1c_1 + 4c_2 &= 7 \\
2c_1 + 5c_2 &= 8 \\
3c_1 + 6c_2 &= 9
\end{aligned}
\]

Solve using row reduction or substitution to find \(c_1\) and \(c_2\).

---

## 4. The Matrix Equation \(A\mathbf{x} = \mathbf{b}\)
The matrix equation represents a system of linear equations compactly:

\[
A\mathbf{x} = \mathbf{b}
\]

where:

- \(A\) is the coefficient matrix.
- \(\mathbf{x}\) is the vector of variables.
- \(\mathbf{b}\) is the vector of constants.

#### Example

\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \mathbf{b} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
\]

\[
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
\]

Solve using matrix operations:

\[
\mathbf{x} = A^{-1}\mathbf{b}
\]

---

## 5. Applications of Linear Systems
1. **Physics**: Modeling forces and motion.
2. **Economics**: Solving input-output models.
3. **Computer Graphics**: Transformations and projections.
4. **Engineering**: Circuit analysis.

---

## 6. Linear Independence
Vectors \(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\) are linearly independent if the only solution to:

\[
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0}
\]

is \(c_1 = c_2 = \cdots = c_n = 0\).

### Example
Are the vectors:

\[
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
\]

linearly independent?

Solution:

\[
c_1\begin{bmatrix} 1 \\ 0 \end{bmatrix} + c_2\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\]

Results in \(c_1 = 0, c_2 = 0\). Hence, the vectors are independent.

---

### Code Example
```python
import numpy as np

# Solve Ax = b
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

x = np.linalg.solve(A, b)
print("Solution:", x)
```
```python exec="on"
import numpy as np

# Solve Ax = b
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

x = np.linalg.solve(A, b)
print("Solution:", x)
```