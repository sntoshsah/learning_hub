# Unsupervised Machine Learning Algorithms

Unsupervised Machine Learning is a type of machine learning where the model learns patterns and structures from unlabeled data. It is primarily used for clustering, dimensionality reduction, association rule mining, and anomaly detection.

---

## Key Concepts

### Unsupervised Learning Objective
Given data \( X = \{x_1, x_2, \dots, x_n\} \), the goal is to learn patterns or structures in the data without explicit output labels.

### Applications
- **Clustering**: Grouping similar data points.
- **Dimensionality Reduction**: Simplifying data while retaining key patterns.
- **Association Rule Mining**: Discovering relationships between variables.
- **Anomaly Detection**: Identifying outliers or abnormal instances.

---

## Types of Unsupervised Algorithms

### 1. Clustering Algorithms
Clustering divides data into groups (clusters) based on similarity.

#### a. **K-Means Clustering**
K-Means partitions data into \( k \) clusters by minimizing intra-cluster variance.

**Mathematical Formulation**:

\[
\text{Objective: } \min \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2
\]

Where:

- \( C_i \): Cluster \( i \).
- \( \mu_i \): Centroid of cluster \( C_i \).

**Algorithm**:

1. Initialize \( k \) centroids.
2. Assign each point to the nearest centroid.
3. Recompute centroids.
4. Repeat until convergence.

**Use Cases**:

- Customer segmentation.
- Image compression.
- Market segmentation.

---

#### b. **Hierarchical Clustering**
Builds a tree-like structure of nested clusters.

**Two Approaches**:

- **Agglomerative**: Bottom-up merging of clusters.
- **Divisive**: Top-down splitting of clusters.

**Linkage Methods**:

- Single Linkage: \( \min \{d(a, b) : a \in A, b \in B\} \)
- Complete Linkage: \( \max \{d(a, b) : a \in A, b \in B\} \)
- Average Linkage: \( \text{mean} \{d(a, b) : a \in A, b \in B\} \)

**Use Cases**:

- Gene expression analysis.
- Social network analysis.

---

#### c. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
Groups points based on density and identifies outliers as noise.

**Parameters**:

- \( \epsilon \): Neighborhood radius.
- \( \text{MinPts} \): Minimum points to form a dense region.

**Use Cases**:

- Geospatial data analysis.
- Noise filtering.

---

### 2. Dimensionality Reduction Algorithms

#### a. **Principal Component Analysis (PCA)**
PCA reduces dimensionality by projecting data onto orthogonal axes that maximize variance.

**Mathematical Formulation**:

Given data matrix \( X \), the principal components are the eigenvectors of the covariance matrix \( \Sigma = \frac{1}{n} X^T X \).

**Use Cases**:

- Data visualization.
- Noise reduction.

---

#### b. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
t-SNE maps high-dimensional data to a lower-dimensional space while preserving local structure.

**Use Cases**:

- Visualizing high-dimensional datasets.
- Exploring clusters.

---

### 3. Association Algorithms

#### a. **Apriori Algorithm**
Discovers frequent itemsets and association rules in transaction data.

**Steps**:
1. Identify frequent itemsets using a minimum support threshold.
2. Generate association rules using a minimum confidence threshold.

**Mathematical Definitions**:

- **Support**:
Proportion of transactions containing an itemset:

\[
\text{Support}(A) = \frac{\text{Transactions with } A}{\text{Total Transactions}}
\]

- **Confidence**: 
Likelihood of \( B \) given \( A \):

\[
\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}
\]

- **Lift**: 
Measures the strength of the rule:

\[
\text{Lift}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \cdot \text{Support}(B)}
\]

**Use Cases**:

- Market basket analysis.
- Recommender systems.

---

#### b. **FP-Growth Algorithm**
Efficiently mines frequent itemsets without candidate generation by using a prefix-tree structure.

**Use Cases**:

- Retail analytics.
- Fraud detection.

---

### 4. Anomaly Detection Algorithms

#### a. **Isolation Forest**
Detects anomalies by isolating instances using a tree structure.

**Key Idea**:

Anomalies are isolated quickly due to their rarity and differences.

**Use Cases**:

- Fraud detection.
- Network intrusion detection.

---

#### b. **One-Class SVM**
Classifies data into one class, treating outliers as anomalies.

**Mathematical Formulation**:

Solves:

\[
\min \frac{1}{2} ||w||^2 + \frac{1}{\nu n} \sum \max(0, 1 - (w \cdot x - \rho))
\]

Where:

- \( \nu \): Fraction of anomalies.

**Use Cases**:

- Manufacturing defect detection.
- Medical diagnosis.

---

#### c. **Elliptic Envelope**
Fits data to a Gaussian distribution and identifies anomalies based on Mahalanobis distance.

**Use Cases**:

- Financial fraud detection.
- Sensor anomaly detection.

---

## Performance Metrics for Unsupervised Learning

### Clustering Metrics

**1. Silhouette Score**:

\[
\text{Silhouette}(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Where:

- \( a(i) \): Average distance within the same cluster.
- \( b(i) \): Average distance to the nearest cluster.


**2. Davies-Bouldin Index**:

\[
DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \frac{s_i + s_j}{d(c_i, c_j)}
\]

Where:

- \( s_i \): Cluster dispersion.
- \( d(c_i, c_j) \): Distance between cluster centroids.

**3. Dunn Index**:

\[
\text{Dunn} = \frac{\min_{i \neq j} \delta(C_i, C_j)}{\max_{1 \leq k \leq k} \Delta(C_k)}
\]

Where:

- \( \delta \): Inter-cluster distance.
- \( \Delta \): Intra-cluster distance.

---

### Anomaly Detection Metrics
1. **Precision**: Proportion of true anomalies among detected anomalies.
2. **Recall**: Proportion of true anomalies detected.
3. **F1-Score**: Harmonic mean of precision and recall.
4. **Area Under ROC Curve (AUC)**: Measures the trade-off between true positive rate and false positive rate.

---

## Conclusion

Unsupervised learning algorithms are powerful tools for exploring data, identifying patterns, and detecting anomalies. Choosing the right algorithm depends on the problem type, data structure, and desired outcome.
