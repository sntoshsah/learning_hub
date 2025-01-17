# Plane and Space Vectors


## 1. Vectors in Terms of Coordinates

### Definition
A vector in 2D or 3D space is represented as an ordered pair or triplet of coordinates:

- **2D vector**: \( \mathbf{v} = (v_x, v_y) \)
- **3D vector**: \( \mathbf{v} = (v_x, v_y, v_z) \)

### Operations
1. **Addition**: \( \mathbf{u} + \mathbf{v} = (u_x + v_x, u_y + v_y) \)
2. **Scalar Multiplication**: \( c \cdot \mathbf{v} = (c v_x, c v_y) \)
3. **Dot Product**:

\[
\mathbf{u} \cdot \mathbf{v} = u_x v_x + u_y v_y + u_z v_z
\]

4. **Cross Product (3D)**:

\[
\mathbf{u} \times \mathbf{v} = (u_y v_z - u_z v_y, u_z v_x - u_x v_z, u_x v_y - u_y v_x)
\]

### Example (Python Code):
```python
import numpy as np
import matplotlib.pyplot as plt

# Vectors
u = np.array([3, 2, 1])
v = np.array([1, -1, 4])

# Vector operations
vector_add = u + v
scalar_mul = 2 * u
dot_product = np.dot(u, v)
cross_product = np.cross(u, v)

print("Vector Addition:", vector_add)
print("Scalar Multiplication:", scalar_mul)
print("Dot Product:", dot_product)
print("Cross Product:", cross_product)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, u[0], u[1], u[2], color='r', label='u')
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='b', label='v')
ax.quiver(0, 0, 0, vector_add[0], vector_add[1], vector_add[2], color='g', label='u+v')
ax.set_xlim([-1, 5])
ax.set_ylim([-2, 3])
ax.set_zlim([0, 5])
ax.legend()
plt.show()
```

---

## 2. Lines and Planes in Space

### Lines in Space
A line in 3D space is defined by a point \( \mathbf{r_0} \) and a direction vector \( \mathbf{v} \):

\[
\mathbf{r}(t) = \mathbf{r_0} + t \mathbf{v}, \quad t \in \mathbb{R}
\]

### Planes in Space
A plane is defined by a point \( \mathbf{r_0} \) and a normal vector \( \mathbf{n} \):

\[
\mathbf{n} \cdot (\mathbf{r} - \mathbf{r_0}) = 0
\]

Expanded:

\[
A x + B y + C z + D = 0
\]

### Example (Python Code):
```python
from mpl_toolkits.mplot3d import Axes3D

# Line parameters
r0 = np.array([1, 2, 3])
v = np.array([2, -1, 1])
t = np.linspace(-5, 5, 100)
line = r0[:, None] + t * v[:, None]

# Plane parameters
x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
A, B, C, D = 2, -1, 3, -4
z = (-A * x - B * y - D) / C

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(line[0], line[1], line[2], label='Line')
ax.plot_surface(x, y, z, alpha=0.5, label='Plane')
ax.legend()
plt.show()
```

---

## 3. Derivatives and Integrals of Vector Functions, Arc Length, and Motion in Space

### Derivatives of Vector Functions
For \( \mathbf{r}(t) = \langle x(t), y(t), z(t) \rangle \):

\[
\mathbf{r}'(t) = \langle x'(t), y'(t), z'(t) \rangle
\]

### Arc Length
\[
L = \int_a^b \| \mathbf{r}'(t) \| dt
\]

### Example (Python Code):
```python
from scipy.integrate import quad

# Vector function
r = lambda t: np.array([np.sin(t), np.cos(t), t])
dr_dt = lambda t: np.array([np.cos(t), -np.sin(t), 1])

# Arc length
arc_length = quad(lambda t: np.linalg.norm(dr_dt(t)), 0, 2 * np.pi)[0]
print("Arc Length:", arc_length)

# Visualization
t_vals = np.linspace(0, 2 * np.pi, 100)
curve = np.array([r(t) for t in t_vals]).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve[0], curve[1], curve[2], label='Curve')
ax.legend()
plt.show()
```

---

## 4. Unit Tangent Vector, Curvature, and TNB System

### Unit Tangent Vector
\[
\mathbf{T}(t) = \frac{\mathbf{r}'(t)}{\| \mathbf{r}'(t) \|}
\]

### Curvature
\[
\kappa(t) = \frac{\| \mathbf{r}'(t) \times \mathbf{r}''(t) \|}{\| \mathbf{r}'(t) \|^3}
\]

### TNB System
- **Tangent (T)**: Direction of motion.
- **Normal (N)**: Perpendicular to \( T \), pointing toward the center of curvature.
- **Binormal (B)**: \( \mathbf{T} \times \mathbf{N} \).

#### Example (Python Code):
```python
# TNB System
r_ddt = lambda t: np.array([-np.sin(t), -np.cos(t), 0])
T = lambda t: dr_dt(t) / np.linalg.norm(dr_dt(t))
N = lambda t: r_ddt(t) / np.linalg.norm(r_ddt(t))
B = lambda t: np.cross(T(t), N(t))

# Compute T, N, B at a specific point
t = np.pi / 4
T_vec = T(t)
N_vec = N(t)
B_vec = B(t)
print("Tangent Vector:", T_vec)
print("Normal Vector:", N_vec)
print("Binormal Vector:", B_vec)

# Visualization
origin = r(t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(*origin, *T_vec, color='r', label='Tangent')
ax.quiver(*origin, *N_vec, color='g', label='Normal')
ax.quiver(*origin, *B_vec, color='b', label='Binormal')
ax.legend()
plt.show()
```