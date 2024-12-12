# Classification

Classification is a supervised learning task that involves predicting the category or class label of a given input. This task is foundational to many applications like spam detection, sentiment analysis, fraud detection, and image classification.

---

## Key Concepts

### Mathematical Formulation
Given a dataset:

\[
D = \{(X_1, y_1), (X_2, y_2), \dots, (X_n, y_n)\}
\]

Where:

- \( X_i \) represents feature vectors.
- \( y_i \) is the class label (\( y_i \in \{C_1, C_2, \dots, C_k\} \)).

The objective is to learn a function \( f \) such that:

\[
f(X) = \hat{y}
\]

Where \( \hat{y} \) is the predicted class label.

---

## Types of Classification Algorithms

### 1. Logistic Regression
Logistic Regression predicts the probability of a binary class using the logistic (sigmoid) function.

#### Mathematical Formula
For binary classification (\( y \in \{0, 1\} \)):

\[
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
\]

Decision rule:

\[
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1|X) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

---

### 2. Decision Tree Classifier
Decision Trees split data based on feature conditions to predict class labels.

#### Splitting Criterion
Common measures:

- **Gini Index**:

\[
G = 1 - \sum_{i=1}^k p_i^2
\]

- **Entropy**:

\[
H = -\sum_{i=1}^k p_i \log_2 p_i
\]

Where \( p_i \) is the probability of class \( i \).

---

### 3. Random Forest Classifier
Random Forest aggregates predictions from multiple decision trees.

#### Prediction Formula
For classification:

\[
\hat{y} = \text{Mode}(\{T_1(X), T_2(X), \dots, T_m(X)\})
\]

Where \( T_i(X) \) is the prediction of the \( i \)-th tree.


---

### 4. Support Vector Machine (SVM)
SVM finds the hyperplane that best separates the classes with the largest margin.

#### Mathematical Formulation
Given:

- Data points \( X_i \)
- Labels \( y_i \in \{-1, 1\} \)

Objective:

\[
\text{Maximize } \frac{2}{||w||}
\]

Subject to:

\[
y_i (w \cdot X_i + b) \geq 1
\]

#### Kernel Trick
For non-linear data, SVM uses kernel functions:

- Linear: \( K(X, X') = X \cdot X' \)
- Polynomial: \( K(X, X') = (X \cdot X' + c)^d \)
- RBF: \( K(X, X') = e^{-\gamma ||X - X'||^2} \)

---

### 5. K-Nearest Neighbors (KNN)
KNN classifies data based on the majority vote of its \( k \)-nearest neighbors.

#### Decision Rule

\[
\hat{y} = \text{Mode}(\{y_{i_1}, y_{i_2}, \dots, y_{i_k}\})
\]

Where \( y_{i_j} \) are the labels of the nearest neighbors.

---

### 6. Naive Bayes Classifier
Naive Bayes applies Bayes' theorem under the assumption of conditional independence.

#### Formula
\[
P(y|X) \propto P(X|y)P(y)
\]

For features \( X = \{x_1, x_2, \dots, x_n\} \):
\[
P(X|y) = \prod_{i=1}^n P(x_i|y)
\]

---

### 7. Neural Networks
Neural Networks use layers of interconnected neurons to model complex patterns.

#### Formula
For a single neuron:

\[
z = w \cdot X + b, \quad a = \sigma(z)
\]

Where:

- \( z \) is the weighted sum.
- \( a \) is the activation output.
- \( \sigma \) is the activation function (e.g., sigmoid, ReLU).

---

### 8. Gradient Boosting Classifier
Gradient Boosting builds an additive model by minimizing a loss function.

#### Update Rule
\[
F_m(X) = F_{m-1}(X) + h_m(X)
\]

Where \( h_m(X) \) is the weak learner (decision tree).

---

## Performance Metrics

### Confusion Matrix
A confusion matrix summarizes prediction results:

|               | Predicted Positive | Predicted Negative |
|---------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP)      | False Negative (FN)      |
| **Actual Negative** | False Positive (FP)     | True Negative (TN)       |

### Metrics
- **Accuracy**:

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

- **Precision**:

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

- **Recall**:

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

- **F1-Score**:

\[
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

- **ROC-AUC**:

Area under the ROC curve measures the model's ability to distinguish between classes.

---

## Choosing the Right Algorithm

- **Linear Relationships**: Logistic Regression, SVM (with linear kernel).
- **Non-linear Data**: Decision Trees, Random Forest, SVM (with RBF kernel).
- **Text Data**: Naive Bayes, Logistic Regression.
- **Large Datasets**: Neural Networks, Gradient Boosting.

---

## Conclusion

Classification is a cornerstone of machine learning with algorithms ranging from simple models like Logistic Regression to complex ones like Gradient Boosting. Selecting the right model depends on the data and problem domain.
