
# Decision Tree Learning Algorithm

## 1. Introduction

The decision tree learning algorithm follows a:

> **Top-down, greedy search strategy using recursive divide-and-conquer**

This approach is used in classical algorithms such as:

- ID3  
- C4.5  
- CART  

---

## 2. High-Level Intuition

A decision tree is built by:

- Starting with the entire dataset  
- Selecting the **best feature to split**  
- Dividing the dataset into subsets  
- Repeating the process recursively  

👉 Goal: Create subsets that are as **pure** as possible (i.e., contain similar class labels)

---

## 3. Core Algorithmic Steps

### Step 1: Initialization

- Start with the **root node**
- It represents the **entire training dataset**

---

### Step 2: Evaluate Attributes

- For each feature (attribute), compute how well it separates the data
- Use statistical measures like:
  - Entropy
  - Information Gain
  - Gini Index

---

### Step 3: Select the Best Attribute

- Choose the attribute that best splits the data
- Typically, this is the one with **maximum Information Gain**

---

### Step 4: Create Branches

- For the selected attribute:
  - Create a branch for each possible value
- Each branch represents a subset of the data

---

### Step 5: Recursive Partitioning

- Split the dataset based on the selected attribute
- Repeat the same process for each subset

---

## 4. Mathematical Selection Criteria

To choose the best split, we measure **impurity**.

---

### 4.1 Entropy

Entropy measures the randomness (impurity) in the dataset.


$$ 
Entropy(S) = - p_+ \log_2(p_+) - p_- \log_2(p_-) 
$$


Where:

* ( $p_+$ ): proportion of positive class
* ( $p_-$ ): proportion of negative class

👉 Lower entropy = more pure dataset

---

### 4.2 Information Gain

Information Gain measures the **reduction in entropy** after a split.

$$
IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
$$

Where:

* ( S ): dataset
* ( A ): attribute
* ( $S_v$ ): subset where attribute ( A = v )

👉 Choose attribute with **highest Information Gain**

---

### 4.3 Gini Index

Used mainly in CART algorithm.

$$
Gini(S) = 1 - \sum_{i=1}^{k} p_i^2
$$

Where:

* ( $p_i$ ): probability of class ( i )

👉 Lower Gini = better split

---

## 5. Stopping Conditions

The recursive splitting stops when:

---

### 5.1 Pure Node

* All samples belong to the same class
* Entropy = 0

👉 Create a **leaf node**

---

### 5.2 No Remaining Attributes

* No features left to split

👉 Use **majority voting** to assign class

---

### 5.3 Empty Dataset

* No samples reach a branch

👉 Assign the **most common class from the parent node**

---

## 6. Key Characteristics

### 6.1 Greedy Nature

* Chooses the best split at each step
* Does **not reconsider previous decisions**

---

### 6.2 Interpretability

* Can be visualized as:

  * A tree structure
  * A set of IF-THEN rules

Example:

```
IF Outlook = Sunny AND Humidity = High → No
ELSE IF Outlook = Rain AND Wind = Weak → Yes
```

---

### 6.3 Invariance to Scaling

* Decision trees do **not require feature scaling**
* Normalization or standardization is unnecessary

---

## 7. Summary

| Component | Description                     |
| --------- | ------------------------------- |
| Strategy  | Greedy, top-down                |
| Process   | Recursive partitioning          |
| Goal      | Maximize purity                 |
| Measures  | Entropy, Information Gain, Gini |
| Output    | Tree or rule-based model        |

---

## 8. Conclusion

Decision trees are:

* Simple and interpretable
* Powerful for classification and regression
* Built using a recursive, greedy splitting strategy

👉 They form the foundation for advanced models like:

* Random Forests
* Gradient Boosted Trees

---
