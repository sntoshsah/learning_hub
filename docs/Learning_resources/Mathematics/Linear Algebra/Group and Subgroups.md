# Introduction to Group and Subgroups

## What is a Group?
A **group** is a set \( G \) equipped with an operation \( * \) that combines any two elements \( a \) and \( b \) to form another element, denoted \( a * b \). To qualify as a group, the set and operation must satisfy four properties:

1. **Closure**: For all \( a, b \in G \), \( a * b \in G \).
2. **Associativity**: For all \( a, b, c \in G \), \( (a * b) * c = a * (b * c) \).
3. **Identity Element**: There exists an element \( e \in G \) such that \( e * a = a * e = a \) for all \( a \in G \).
4. **Inverse Element**: For each \( a \in G \), there exists \( b \in G \) such that \( a * b = b * a = e \).

### Examples
1. **Integers under Addition**:
   - Set: \( \mathbb{Z} \)
   - Operation: Addition (+)
   - Properties: Satisfies closure, associativity, identity (0), and inverses (-a).

2. **Non-Zero Real Numbers under Multiplication**:
   - Set: \( \mathbb{R}^* \)
   - Operation: Multiplication (×)
   - Properties: Satisfies closure, associativity, identity (1), and inverses (\( 1/a \)).

---

## Sets and Set Operations

### Definition of a Set
A **set** is a well-defined collection of distinct objects, called elements. Examples:

\[
A = \{1, 2, 3, 4\}, \quad B = \{x \in \mathbb{R} : x^2 \leq 4\}
\]

## Common Set Operations

### Union

\[
A \cup B = \{x : x \in A \text{ or } x \in B\}
\]

Example:
\( A = \{1, 2\}, B = \{2, 3\} \Rightarrow A \cup B = \{1, 2, 3\} \).

### Intersection

\[
A \cap B = \{x : x \in A \text{ and } x \in B\}
\]

Example:
\( A = \{1, 2\}, B = \{2, 3\} \Rightarrow A \cap B = \{2\} \).

### Difference

\[
A \setminus B = \{x : x \in A \text{ and } x \notin B\}
\]

Example:
\( A = \{1, 2\}, B = \{2, 3\} \Rightarrow A \setminus B = \{1\} \).

### Cartesian Product

\[
A \times B = \{(a, b) : a \in A, b \in B\}
\]

Example:
\( A = \{1, 2\}, B = \{3, 4\} \Rightarrow A \times B = \{(1, 3), (1, 4), (2, 3), (2, 4)\} \).

---

## Mappings

### Definition
A **mapping** (or function) is a rule that assigns each element of one set (domain) to exactly one element of another set (codomain).

\[
 f : A \to B \quad \text{where } f(a) \in B \text{ for all } a \in A.
\]

### Types of Mappings
1. **Injective (One-to-One)**: Different elements in the domain map to different elements in the codomain.
2. **Surjective (Onto)**: Every element in the codomain is the image of some element in the domain.
3. **Bijective**: Both injective and surjective.

### Example
\( f(x) = x^2 \):

- Domain: \( \mathbb{R} \), Codomain: \( \mathbb{R} \)
- Not injective (\( f(-2) = f(2) \)).
- Not surjective (no \( x \) maps to \( -1 \) in \( \mathbb{R} \)).

---

## Group

### Examples of Groups

### Symmetry Group
The set of all rotations and reflections of a geometric figure forms a group under composition.

### Modular Arithmetic
For \( n \in \mathbb{Z} \), the set \( \{0, 1, \dots, n-1\} \) under addition modulo \( n \) is a group.

---

## Subgroups

### Definition
A subset \( H \subseteq G \) is a **subgroup** of \( G \) if \( H \) itself forms a group under the operation of \( G \).

### Subgroup Criteria
1. \( e \in H \).
2. For all \( a, b \in H \), \( a * b \in H \).
3. For all \( a \in H \), \( a^{-1} \in H \).

### Example
\( G = \mathbb{Z}, H = 2\mathbb{Z} \):

1. \( 0 \in H \).
2. Closure: Sum of even numbers is even.
3. Inverses: Negatives of even numbers are even.

---

# Tips and Tricks

1. **Verify Group Properties**: Always check all four properties to confirm a group.
2. **Subgroup Test**: For finite groups, closure and inverses imply the subgroup.
3. **Symmetry Insight**: Symmetry groups simplify problems in geometry and physics.


