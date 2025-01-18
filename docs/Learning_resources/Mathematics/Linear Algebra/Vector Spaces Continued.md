# Vector Spaces Continued
## The Dimension of a Vector Space

### Definition
The **dimension** of a vector space is the number of vectors in its basis, which is a linearly independent set that spans the entire space.

### Formula
If \( V \) is a vector space with a basis \( \{v_1, v_2, \ldots, v_n\} \), then:

\[
\dim(V) = n
\]

### Examples
1. The dimension of \( \mathbb{R}^3 \) is 3.
2. For the space of polynomials of degree \( \leq 2 \), the dimension is 3 (basis: \( \{1, x, x^2\} \)).

### Tips and Tricks
1. To find the dimension of a vector space, determine the maximum number of linearly independent vectors.
2. Use the row-reduction method to identify a basis from a set of vectors.
3. The number of pivot columns in the row echelon form equals the dimension of the column space.

---

## Rank

### Definition
The **rank** of a matrix is the dimension of its column space (or row space), representing the maximum number of linearly independent columns (or rows).

### Formula
\[
\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A))
\]

### Examples
1. For \(
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
\):
   - Row-reduce to find the rank is 2.

2. For \(
B = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
\):
   - Row-reduce to find the rank is 3.

### Tips and Tricks
1. A square matrix is invertible if and only if \( \text{rank}(A) = n \), where \( n \) is the size of the matrix.
2. Use the singular value decomposition (SVD) to numerically compute the rank for large matrices.

---

## Change of Basis

### Definition
Changing the basis of a vector space means expressing vectors or transformations in terms of a different basis.

### Transformation Formula
If \( B \) and \( B' \) are bases of \( V \), the change of basis matrix \( P \) satisfies:

\[
[P]_B^{B'} v_B = v_{B'}
\]

### Steps to Change Basis
1. **Construct the Change of Basis Matrix**: 
   - Write the vectors of the new basis \( B' \) as columns in terms of the old basis \( B \).

2. **Transform Coordinates**: 
   - Multiply the change of basis matrix with the coordinate vector in the old basis to get the vector in the new basis.

### Example
Transform coordinates from the standard basis \( B_s = \{(1, 0), (0, 1)\} \) to \( B = \{(1, 1), (1, -1)\} \).

#### Solution
1. Write \( B \) in terms of \( B_s \):
\[
P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\]

2. For a vector \( v = \begin{bmatrix} 3 \\ 1 \end{bmatrix} \), compute:\

\[
P^{-1} v = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \end{bmatrix}
\]

---

## Applications of Difference Equations

### Definition
Difference equations describe the relationship between consecutive terms in a sequence, often representing discrete systems.

### Example
\( x_{n+1} = 3x_n - 4x_{n-1} \).

#### Steps to Solve
1. Find the characteristic equation: \( r^2 - 3r + 4 = 0 \).
2. Solve for roots. If roots are real and distinct, the solution is:

\[
x_n = C_1 r_1^n + C_2 r_2^n
\]

### Applications
1. **Population Modeling**: Predict population growth or decline over discrete time intervals.
2. **Financial Calculations**: Compute compound interest, loan payments, etc.
3. **Signal Processing**: Analyze discrete signals in time-series data.

### Example Problem
Given \( x_{n+1} = 2x_n - x_{n-1} \), solve for \( x_n \).

#### Solution
1. Characteristic equation: \( r^2 - 2r + 1 = 0 \).
2. Roots: \( r = 1 \) (repeated root).
3. General solution:

\[
x_n = C_1 + C_2 n
\]

---

## Applications of Markov Chains

### Definition
A **Markov chain** is a stochastic process with memoryless transitions between states, meaning the probability of transitioning to the next state depends only on the current state.

### Transition Matrix
The probabilities of moving from one state to another are represented in a matrix \( P \):

\[
P = \begin{bmatrix} p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn} \end{bmatrix}
\]

where \( p_{ij} \) is the probability of transitioning from state \( i \) to state \( j \).

### Applications
1. **Google PageRank**: Determines the importance of webpages based on link structure.
2. **Weather Prediction**: Models probabilities of weather transitions (e.g., sunny to rainy).
3. **Queueing Systems**: Analyzes customer arrival and service processes.

### Example
A Markov chain with states \( A \) and \( B \):

\[
P = \begin{bmatrix} 0.7 & 0.3 \\
0.4 & 0.6 \end{bmatrix}
\]

#### Problem
Find the steady-state distribution.

#### Solution
1. Solve \( \pi P = \pi \):

\[
\begin{bmatrix} \pi_A & \pi_B \end{bmatrix} \begin{bmatrix} 0.7 & 0.3 \\
0.4 & 0.6 \end{bmatrix} = \begin{bmatrix} \pi_A & \pi_B \end{bmatrix}
\]

2. Solve the system of equations:

\[
0.7\pi_A + 0.4\pi_B = \pi_A \\
0.3\pi_A + 0.6\pi_B = \pi_B \\
\pi_A + \pi_B = 1
\]

3. Solution: \( \pi_A = 0.571, \pi_B = 0.429 \).

