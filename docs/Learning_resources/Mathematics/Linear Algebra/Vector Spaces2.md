---
title: Vector Spaces
description: A comprehensive guide to understanding vector spaces, their key concepts, mathematical foundations, applications, and exercises with solutions.
---

# Vector Spaces

## 1. Vector Space and Subspace

### Introduction
A **vector space** is a collection of objects called vectors, which can be added together and multiplied by scalars while satisfying specific axioms. Examples include Euclidean space \( \mathbb{R}^n \), spaces of functions, and polynomial spaces.

A **subspace** is a subset of a vector space that is also a vector space under the same operations.

#### Why are Vector Spaces Important?
- Form the foundation of linear algebra.
- Used in diverse fields such as physics, computer graphics, and data science.

### Key Concepts
1. **Vector Space Axioms**:
    - Closure under addition and scalar multiplication.
    - Associativity and commutativity of addition.
    - Existence of additive identity and additive inverses.
    - Compatibility of scalar multiplication.

2. **Subspaces**:
    - Must contain the zero vector.
    - Closed under vector addition and scalar multiplication.

#### Example
Let \( V = \mathbb{R}^3 \) and consider \( W = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\} \). Prove \( W \) is a subspace.

**Solution:**

1. **Zero Vector**: \( (0, 0, 0) \in W \).
2. **Closure under Addition**: 
If \( (x_1, y_1, z_1), (x_2, y_2, z_2) \in W \), then:

    \[ (x_1 + x_2) + (y_1 + y_2) + (z_1 + z_2) = 0 \]

3. **Closure under Scalar Multiplication**: For \( k \in \mathbb{R} \):
\[ kx + ky + kz = k(x + y + z) = 0 \]

Thus, \( W \) is a subspace.

---

## 2. Null Space and Column Space, and Linear Transformations

### Null Space and Column Space
1. **Null Space (Kernel)**:
   The set of all solutions to \( A\mathbf{x} = \mathbf{0} \).

2. **Column Space (Range)**:
   The span of the columns of matrix \( A \).

#### Example
Find the null space and column space of:

\[ A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \]

**Solution:**

- **Null Space**: Solve \( A\mathbf{x} = \mathbf{0} \):

\[ \mathbf{x} = \begin{bmatrix} t \\ -2t \\ t \end{bmatrix}, \quad t \in \mathbb{R} \]

Basis: \( \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \).

- **Column Space**: Rank of \( A = 2 \). 
Basis vectors:

  	\[ \begin{bmatrix} 1 \\ 4 \\ 7 \end{bmatrix}, \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} \]

### Linear Transformations
A function \( T: V \to W \) is linear if:

- \( T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) \).
- \( T(c\mathbf{u}) = cT(\mathbf{u}) \).

#### Example
Show \( T(x, y) = (2x, 3y) \) is a linear transformation.

**Solution:**

- **Addition**: \( T((x_1, y_1) + (x_2, y_2)) = T(x_1 + x_2, y_1 + y_2) = (2(x_1 + x_2), 3(y_1 + y_2)) \).
- **Scalar Multiplication**: \( T(c(x, y)) = T(cx, cy) = (2cx, 3cy) \).

---

## 3. Linearly Independent Sets, Bases

### Linearly Independent Sets
A set of vectors is linearly independent if no vector can be written as a linear combination of the others.

#### Example
Determine if \( \{(1, 0), (0, 1), (1, 1)\} \) is linearly independent.

**Solution:**
Solve:

\[ c_1(1, 0) + c_2(0, 1) + c_3(1, 1) = (0, 0) \]

This gives \( c_1 + c_3 = 0 \) and \( c_2 + c_3 = 0 \). Only solution: \( c_1 = c_2 = c_3 = 0 \). Independent.

### Bases
A basis of a vector space is a linearly independent set that spans the space.

#### Example
Find a basis for \( \mathbb{R}^2 \).

**Solution:**
Standard basis: \( \{(1, 0), (0, 1)\} \).

---

## 4. Coordinate System

### Introduction
Coordinates represent a vector relative to a basis.

#### Example
Given basis \( \{(1, 2), (3, 4)\} \), find coordinates of \( \mathbf{v} = (7, 10) \).

**Solution:**
Solve \( c_1(1, 2) + c_2(3, 4) = (7, 10) \):

\[ c_1 = 2, \, c_2 = 1 \]
Coordinates: \( (2, 1) \).

---

## Exercises

### Exercise 1
Prove \( W = \{(x, y, z) : x + 2y - z = 0\} \) is a subspace of \( \mathbb{R}^3 \).

**Solution:**

1. **Zero Vector**: \( (0, 0, 0) \in W \).
2. **Addition**: Closure under addition holds.
3. **Scalar Multiplication**: Closure under scalar multiplication holds.

### Exercise 2
Find the null space of \( A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} \).

**Solution:**

Solve \( A\mathbf{x} = \mathbf{0} \):

\[ \mathbf{x} = \begin{bmatrix} t \\ -\frac{1}{2}t \end{bmatrix}, \, t \in \mathbb{R} \]

Basis: \( \begin{bmatrix} 1 \\ -\frac{1}{2} \end{bmatrix} \).

### Exercise 3
Find the basis for the column space of:

\[ A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \]

**Solution:**

Rank: 2.

Basis vectors: 

\[ \begin{bmatrix} 1 \\ 4 \\ 7 \end{bmatrix}, \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} \]
