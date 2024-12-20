# Hypothesis Testing with Proportions and Chi-Square Tests

This repository is a comprehensive guide to hypothesis testing, focusing on proportion tests and chi-square tests. The guide provides theoretical explanations, mathematical formulas, and Python code examples for practical applications.

---

## Topics Covered

1. **One-Sample Proportion Test**:
   - Z-test, Z-score, and p-value.
2. **Two-Sample Proportion Test**.
3. **Chi-Square Test**:
   - Test of independence.
   - Goodness-of-fit test.

---

## 1. One-Sample Proportion Test

### Overview
The one-sample proportion test is used to determine whether the proportion of a specific category in a population matches a hypothesized proportion (\(p_0\)).

**Example**: Testing if 60% of voters support a candidate.

### Hypotheses
- **Null Hypothesis (\(H_0\))**: The population proportion equals \(p_0\).

\[
H_0: p = p_0
\]

- **Alternative Hypothesis (\(H_a\))**: The population proportion differs from \(p_0\).

\[
H_a: p \neq p_0
\]

### Formula
The Z-statistic is calculated as:

\[
Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1 - p_0)}{n}}}
\]

Where:

- \(\hat{p}\): Sample proportion.
- \(p_0\): Hypothesized proportion.
- \(n\): Sample size.

### Python Code Example
```python
from statsmodels.stats.proportion import proportions_ztest

# Data
successes = 48  # Number of successes
n = 100         # Sample size
p0 = 0.6        # Hypothesized proportion

# Perform the test
stat, p_value = proportions_ztest(count=successes, nobs=n, value=p0)
print(f"Z-Statistic: {stat}, p-value: {p_value}")
```

---

## 2. Two-Sample Proportion Test

### Overview
The two-sample proportion test compares the proportions of two independent groups to determine if they are significantly different.

**Example**: Comparing the proportion of male and female students passing an exam.

### Hypotheses
- **Null Hypothesis (\(H_0\))**: The proportions are equal.

\[
H_0: p_1 = p_2
\]

- **Alternative Hypothesis (\(H_a\))**: The proportions are not equal.

\[
H_a: p_1 \neq p_2
\]

### Formula
The Z-statistic is calculated as:

\[
Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{p (1 - p) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
\]

Where:

- \(\hat{p}_1, \hat{p}_2\): Sample proportions.
- \(n_1, n_2\): Sample sizes.
- \(p\): Pooled proportion:
  \[
  p = \frac{x_1 + x_2}{n_1 + n_2}
  \]

### Python Code Example
```python
# Data
successes = [50, 45]  # Successes in group 1 and group 2
sample_sizes = [100, 100]  # Sample sizes of group 1 and group 2

# Perform the test
stat, p_value = proportions_ztest(count=successes, nobs=sample_sizes)
print(f"Z-Statistic: {stat}, p-value: {p_value}")
```

---

## 3. Chi-Square Test

### Overview
Chi-Square tests are used for categorical data to test relationships between variables or whether observed frequencies match expected frequencies.

---

### 3.1 Chi-Square Test of Independence

**Purpose**: To test whether two categorical variables are independent.

**Example**: Testing if gender and preference for a product are related.

### Hypotheses
- **Null Hypothesis (\(H_0\))**: The variables are independent.
- **Alternative Hypothesis (\(H_a\))**: The variables are not independent.

### Formula

\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]

Where:

- \(O\): Observed frequency.
- \(E\): Expected frequency:

    \[
    E = \frac{\text{row total} \times \text{column total}}{\text{grand total}}
    \]

### Python Code Example
```python
import numpy as np
from scipy.stats import chi2_contingency

# Data (contingency table)
data = np.array([[50, 30], [20, 100]])

# Perform the test
chi2, p_value, dof, expected = chi2_contingency(data)
print(f"Chi2-Statistic: {chi2}, p-value: {p_value}")
print("Expected Frequencies:\n", expected)
```

---

### 3.2 Chi-Square Goodness-of-Fit Test

**Purpose**: To test if an observed distribution matches an expected distribution.

**Example**: Testing if the distribution of dice rolls matches a uniform distribution.

### Hypotheses
- **Null Hypothesis (\(H_0\))**: The observed frequencies match the expected frequencies.
- **Alternative Hypothesis (\(H_a\))**: The observed frequencies do not match.

### Formula

\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]

Where:

- \(O\): Observed frequency.
- \(E\): Expected frequency.

### Python Code Example
```python
from scipy.stats import chisquare

# Data
observed = [18, 22, 20, 25, 15]
expected = [20, 20, 20, 20, 20]  # Uniform distribution

# Perform the test
chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
print(f"Chi2-Statistic: {chi2}, p-value: {p_value}")
```

---

## Conclusion

This guide provides a theoretical and practical understanding of hypothesis testing for proportions and categorical data using Chi-Square tests. The Python examples demonstrate how to implement these tests with dummy data. 
