# Logical Interpretation

A decision tree can also be viewed as a **logical expression**, not just a geometric model.

---

## 1. Key Idea

A decision tree represents:

👉 **OR of AND conditions**

* Each **path** from root → leaf = **AND (conjunction)**
* The whole tree = **OR (disjunction)** of all such paths

---

## 2. What is a Conjunction (AND)?

A conjunction means:

```text
Condition1 AND Condition2 AND Condition3
```

Example:

```text
Outlook = Sunny AND Humidity = High
```

👉 This corresponds to **one path in the tree**

---

## 3. What is a Disjunction (OR)?

A disjunction means:

```text
Path1 OR Path2 OR Path3
```

👉 This combines **multiple paths**

---

## 4. Example Tree Logic

Imagine a simple decision tree:

```
IF Outlook = Sunny AND Humidity = High → No
IF Outlook = Rain AND Wind = Weak → Yes
```

This can be written as:

$$
(Outlook = \text{Sunny} \land Humidity = \text{High})
\lor
(Outlook = \text{Rain} \land Wind = \text{Weak})
$$

👉 This is called **Disjunctive Normal Form (DNF)**

---

## 5. Why this matters

* Helps convert trees → **rules**
* Makes models **interpretable**
* Useful in **expert systems & rule engines**

---


### Logical Form

$$
(Outlook = Sunny \land Humidity = High)
\lor
(Outlook = Rain \land Wind = Weak)
$$
---

## 5. General Mathematical Form

For a tree with k leaf nodes:

$$
f(x) = \bigvee_{i=1}^{k} \left( \bigwedge_{j=1}^{m_i} condition_{ij} \right)
$$

Where:

* ( i ) = path index
* ( j ) = condition index in a path
* ( m_i ) = number of conditions in path ( i )

---

## 6. Interpretation

* Each conjunction defines a **region of the input space**
* The disjunction combines all regions that lead to the same prediction

---

## 7. Advantages of Logical Representation

* Easy to interpret
* Can be converted into **if-then rules**
* Useful for **rule-based systems**
* Helps in explaining model decisions

---

## 8. Connection to Geometry

* Logical AND → Intersection of regions
* Logical OR → Union of regions

---

## 9. Summary

| Concept        | Meaning                 |
| -------------- | ----------------------- |
| Path           | Conjunction (AND)       |
| Tree           | Disjunction (OR)        |
| Representation | Disjunctive Normal Form |

---

## 10. Conclusion

A decision tree can be viewed as a set of logical rules:

> A collection of AND conditions combined using OR

This makes decision trees one of the most **interpretable machine learning models**.

