# Regression Analysis

Regression analysis is a fundamental statistical and machine learning technique for modeling the relationship between a dependent variable and one or more independent variables. The primary goal of regression is to predict or estimate the value of the dependent variable based on the input features.

## Key Concepts

### Mathematical Formulation
The general form of a regression model is:

$$y = f(X) + \epsilon$$

Where:
- \( y \) is the dependent variable.<br>
- \( X \) represents the independent variables.<br>
- \( f(X) \) is the function capturing the relationship between \( X \) and \( y \).<br>
- \( \epsilon \) is the error term (unexplained variation).

For linear regression, \( f(X) \) is linear, while for other methods like polynomial regression, \( f(X) \) may have a more complex form.

---

## Types of Regression Algorithms

### 1. Linear Regression
Linear regression establishes a linear relationship between the dependent variable (\( y \)) and the independent variables (\( X \)).

#### Formula:
\[
y = \beta_0 + \beta_1 X + \epsilon
\]

Where:
- \( \beta_0 \) is the intercept.
- \( \beta_1 \) is the coefficient.
- \( \epsilon \) is the error term.

#### Key Properties:
- Simple and interpretable.
- Prone to underfitting when the relationship is non-linear.

---

### 2. Logistic Regression
Logistic regression predicts the probability of a binary outcome by applying a sigmoid function.

#### Formula:
\[
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
\]

#### Key Properties:
- Suitable for classification tasks.
- Outputs probabilities between 0 and 1.

---

### 3. Polynomial Regression
Polynomial regression extends linear regression by modeling the relationship as an \( n \)-degree polynomial.

#### Formula:
\[
y = \beta_0 + \beta_1 X + \beta_2 X^2 + \dots + \beta_n X^n + \epsilon
\]

#### Key Properties:
- Fits non-linear relationships.
- May overfit if the degree of the polynomial is too high.

---

### 4. Support Vector Regression (SVR)
SVR aims to find a hyperplane that maximizes the margin within a certain tolerance (\( \epsilon \)).

#### Key Concepts:
- Uses kernel functions (linear, polynomial, RBF).
- Robust to outliers.

---

### 5. Decision Tree Regression
Decision Tree Regression uses a tree-like structure to model decisions based on feature splits.

#### Key Properties:
- Non-parametric and interpretable.
- Prone to overfitting without pruning.

---

### 6. Random Forest Regression
Random Forest Regression combines multiple decision trees using ensemble learning to improve accuracy and reduce overfitting.

#### Key Properties:
- Robust to overfitting and noise.
- Handles non-linear relationships.

---

### 7. Ridge Regression
Ridge regression adds an \( L2 \)-regularization term to linear regression to prevent overfitting.

#### Formula:
\[
\text{Minimize: } \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
\]

Where \( \lambda \) controls the regularization strength.

---

### 8. Lasso Regression
Lasso regression uses \( L1 \)-regularization to enforce sparsity in the model, shrinking some coefficients to zero.

#### Formula:
\[
\text{Minimize: } \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
\]

#### Key Properties:
- Performs feature selection.
- Robust to high-dimensional datasets.

---

## Additional Algorithms

### Elastic Net Regression
Elastic Net combines \( L1 \) and \( L2 \)-regularization for better generalization.

#### Formula:
\[
\text{Minimize: } \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
\]

---

### Bayesian Regression
Bayesian regression incorporates prior distributions over model parameters, providing probabilistic predictions.

#### Formula:
\[
p(\beta|X, y) \propto p(y|X, \beta) \cdot p(\beta)
\]

---

### Gradient Boosting Regression
Gradient Boosting builds an additive model using decision trees to minimize a loss function.

#### Key Properties:
- Highly accurate.
- Requires careful tuning to avoid overfitting.

---


## Choosing the Right Algorithm

The choice of regression algorithm depends on:

- **Data Complexity**: Non-linear relationships or noise.
- **Regularization Needs**: High-dimensional or sparse data.
- **Interpretability**: Simpler models like linear regression are easier to explain.

---


## Performance Metrics for Regression

### 1. Mean Absolute Error (MAE)
Measures the average magnitude of errors in predictions:

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
\]

- **Interpretation**: Lower values indicate better model performance.
- **Sensitivity**: Does not penalize large errors as heavily as squared metrics.

---

### 2. Mean Squared Error (MSE)
Measures the average squared difference between actual and predicted values:

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

- **Interpretation**: Heavily penalizes larger errors.
- **Use Case**: When large errors are particularly undesirable.

---

### 3. Root Mean Squared Error (RMSE)
The square root of MSE, providing error in the same units as the target variable:

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

- **Interpretation**: Easier to interpret than MSE due to units.

---

### 4. R-squared (\( R^2 \))
Represents the proportion of variance in the target variable explained by the model:

\[
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
\]

- **Interpretation**:

  - \( R^2 = 1 \): Perfect fit.
  - \( R^2 = 0 \): Model explains no variance.
  - \( R^2 < 0 \): Model is worse than a horizontal mean line.

---

### 5. Adjusted R-squared
Accounts for the number of predictors in the model:

\[
\text{Adjusted } R^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}
\]

Where:

- \( n \): Number of data points.
- \( p \): Number of predictors.

- **Interpretation**: Penalizes adding irrelevant predictors to the model.

---

### 6. Mean Absolute Percentage Error (MAPE)
Expresses error as a percentage of the actual values:

\[
\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
\]

- **Use Case**: Useful when target values vary significantly.

---

### 7. Explained Variance Score
Measures the proportion of variance explained by the model:

\[
\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
\]

- **Interpretation**: Similar to \( R^2 \), but focuses on explained variance.

---

### 8. Huber Loss
Combines MAE and MSE for robust error measurement:

\[
L_{\delta}(a) =
\begin{cases} 
\frac{1}{2}(a)^2 & \text{for } |a| \leq \delta, \\
\delta \cdot (|a| - \frac{\delta}{2}) & \text{for } |a| > \delta
\end{cases}
\]

Where \( a = y - \hat{y} \).

- **Use Case**: Robust to outliers in regression.

---

## Comparison of Metrics

| Metric         | Pros                                         | Cons                                      |
|----------------|---------------------------------------------|------------------------------------------|
| **MAE**        | Easy to interpret, less sensitive to outliers. | May under-penalize large errors.         |
| **MSE**        | Penalizes large errors heavily.              | Less interpretable due to squaring.      |
| **RMSE**       | Same units as target variable.               | Still sensitive to outliers.             |
| **R-squared**  | Shows proportion of variance explained.      | Can be misleading with irrelevant features. |
| **MAPE**       | Provides percentage error.                   | Undefined for \( y_i = 0 \).             |
| **Huber Loss** | Robust to outliers.                         | Requires hyperparameter \( \delta \).    |

---

## Choosing the Right Metric
- **Small datasets with few outliers**: Use MAE or MSE.
- **Sensitive to large errors**: Use RMSE or Huber Loss.
- **Comparing model fit**: Use \( R^2 \) or Adjusted \( R^2 \).
- **Relative error**: Use MAPE for interpretability.

---

## Visual Evaluation Techniques
- **Residual Plots**: Show the difference between actual and predicted values.
- **Parity Plots**: Compare predicted vs. actual values directly.
- **Error Histograms**: Visualize the distribution of errors.

---

## Conclusion

Regression analysis is a versatile tool for predictive modeling. Each algorithm has its strengths and weaknesses, making it crucial to evaluate their performance based on the dataset and task at hand.

---
