# Geometric Interpretation of Decision Trees

## 1. Introduction

A **Decision Tree** can be understood geometrically as a method of **partitioning the feature space** into smaller regions.

* Each split divides the space
* Each region corresponds to a **leaf node**
* Each leaf predicts a constant value (class or regression output)

---

## 2. Feature Space Representation

The feature space depends on the number of features:

| Features   | Geometry            |
| ---------- | ------------------- |
| 1 feature  | Line                |
| 2 features | 2D Plane            |
| 3 features | 3D Space            |
| n features | n-dimensional space |

Each data point is a coordinate:

```math
x = (x_1, x_2, x_3, ..., x_n)
```

---

## 3. Axis-Parallel Splits

### Definition

A standard decision tree uses splits of the form:

```math
x_i > a
```

---

### Geometric Interpretation

* In **2D** → vertical or horizontal line
* In **3D** → plane
* In **nD** → hyperplane

These are always **parallel to coordinate axes**.

---

### Example (2D)

Features:

* ( x_1 ): Age
* ( x_2 ): Income

Split:

```math
x_1 > 30
```

👉 Creates a **vertical boundary** at ( x_1 = 30 )

---

## 4. Hierarchical Partitioning

Decision trees split **recursively**:

1. First split divides space into 2 regions
2. Next split divides one of those regions
3. Process continues until stopping condition

---

### Visualization Idea

```
Step 1: Whole space
Step 2: Split → 2 regions
Step 3: Split → 4 regions
Step 4: Split → more refined regions
```

---

## 5. Resulting Regions: Hyperrectangles

Because splits are axis-aligned:

| Dimension | Shape           |
| --------- | --------------- |
| 2D        | Rectangles      |
| 3D        | Cuboids         |
| nD        | Hyperrectangles |

👉 Each leaf node corresponds to one **hyperrectangle**

---

## 6. Multivariate (Oblique) Splits

Instead of splitting on a single feature:

```math
w_1 x_1 + w_2 x_2 + \dots + w_n x_n > b
```

---

### Geometric Interpretation

* Produces **slanted (oblique) boundaries**
* Not restricted to axes

---

### Resulting Shapes

| Split Type    | Shape      |
| ------------- | ---------- |
| Axis-parallel | Rectangles |
| Multivariate  | Polyhedra  |

---

## 7. 2D Visualization Concept

### Step-by-step splitting

1. First split:

   ```
   x1 > 0.5 → vertical line
   ```

2. Second split:

   ```
   x2 > 0.5 → horizontal line
   ```

3. Third split:

   ```
   x1 > 0.75 → refine region
   ```

👉 Final result: multiple rectangular regions

---

## 8. 3D Visualization Concept

In 3D, splits become planes:

* ( x_1 = 0.5 ) → vertical plane
* ( x_2 = 0.5 ) → horizontal plane

These planes divide space into **3D boxes (cuboids)**

---

## 9. Why This Matters

### Limitations

* Cannot create diagonal boundaries easily
* Needs many splits for complex shapes

---

### Strengths

* Easy to interpret
* Fast to compute
* Works well with tabular data

---

## 10. Key Intuition

> A decision tree divides space into simple regions and assigns a prediction to each region.

---

## 11. Comparison Summary

| Property         | Axis-Parallel Tree | Multivariate Tree  |
| ---------------- | ------------------ | ------------------ |
| Split type       | Single feature     | Linear combination |
| Boundary         | Parallel to axes   | Oblique            |
| Region shape     | Rectangles         | Polyhedra          |
| Interpretability | High               | Lower              |
| Flexibility      | Moderate           | High               |

---

## 12. Optional: Python Visualization Snippet

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(100, 2)

plt.scatter(X[:,0], X[:,1])

# Example splits
plt.axvline(x=0.5)
plt.axhline(y=0.5)

plt.title("Decision Tree Partitioning")
plt.show()
```

---

## 13. Conclusion

Decision trees are best understood as:

* A **geometric partitioning algorithm**
* That divides space into **non-overlapping regions**
* Using **simple, interpretable rules**
