# Hypothesis Testing: Non-Parametric


---

## 1. Assumptions in Hypothesis Testing

### Overview
Assumptions ensure that the results of hypothesis tests are valid and reliable. Violations of these assumptions can lead to incorrect conclusions.

---

### 1.1 Randomness
- **Definition**: Data must be collected randomly to avoid bias.
- **Reason**: Randomness ensures that the sample represents the population.
- **Example**: A random sample of voters provides an unbiased estimate of election preferences.

---

### 1.2 Independence of Observations
- **Definition**: Each observation must be independent, meaning the occurrence of one observation should not influence another.
- **Reason**: Dependence introduces bias and invalidates statistical tests.
- **Example**: Measuring blood pressure in a single individual multiple times violates independence.

---

### 1.3 Large Sample Size
- The Central Limit Theorem (CLT) ensures that the sampling distribution of the mean becomes approximately normal with a sufficiently large sample size (\(n > 30\)).

#### t-Tests
- Large sample size is needed when population variance is unknown.

**Formula**:

\[
t = \frac{\bar{x} - \mu}{s / \sqrt{n}}
\]

**Python Example**:
```python
from scipy.stats import ttest_1samp
import numpy as np

# Simulated data
data = np.random.normal(loc=50, scale=5, size=40)
stat, p_value = ttest_1samp(data, popmean=50)
print(f"t-statistic: {stat}, p-value: {p_value}")
```

#### Proportion Tests

- \(np \geq 5\) and \(n(1-p) \geq 5\) ensure the normal approximation of the binomial distribution.

**Formula**:

\[
Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1-p_0)}{n}}}
\]

**Python Example**:
```python
from statsmodels.stats.proportion import proportions_ztest

successes = 48  # Number of successes
n = 100         # Sample size
p0 = 0.5        # Hypothesized proportion

stat, p_value = proportions_ztest(count=successes, nobs=n, value=p0)
print(f"Z-statistic: {stat}, p-value: {p_value}")
```

#### Chi-Square Tests

- Each expected frequency should be \(\geq 5\).

**Formula**:

\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]

**Python Example**:
```python
import numpy as np
from scipy.stats import chi2_contingency

data = np.array([[50, 30], [20, 100]])
chi2, p_value, dof, expected = chi2_contingency(data)
print(f"Chi2-statistic: {chi2}, p-value: {p_value}")
```

---

## 2. Parametric vs. Non-Parametric Tests

### Parametric Tests
- **Definition**: Tests that assume the data follows a specific distribution (e.g., normal distribution).
- **Examples**: t-tests, ANOVA.
- **Advantages**: More powerful when assumptions are met.
- **Disadvantages**: Sensitive to violations of assumptions.

### Non-Parametric Tests
- **Definition**: Tests that do not assume a specific data distribution.
- **Examples**: Wilcoxon tests, Kruskal-Wallis test.
- **Advantages**: Robust to non-normal data and outliers.
- **Disadvantages**: Less powerful than parametric tests for normal data.

---

## 3. Non-Parametric Tests

### 3.1 Wilcoxon Signed-Rank Test
- **Purpose**: Tests whether the median of paired data differs from a hypothesized value.
- **Hypotheses**:
  - \(H_0\): The medians are equal.
  - \(H_a\): The medians are not equal.

**Mathematical Formula**:

1. Compute differences (\(d_i\)) between paired samples.
2. Rank absolute differences.
3. Calculate the sum of signed ranks (\(W\)).

**Python Example**:
```python
from scipy.stats import wilcoxon

# Paired data
before = [88, 85, 90, 92, 91]
after = [84, 82, 88, 89, 86]

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(before, after)
print(f"Statistic: {stat}, p-value: {p_value}")
```

---

### 3.2 Wilcoxon-Mann-Whitney Test
- **Purpose**: Compares medians of two independent samples.
- **Hypotheses**:

	- \(H_0\): The distributions are the same.
	- \(H_a\): The distributions are different.

**Python Example**:
```python
from scipy.stats import mannwhitneyu

# Two independent samples
group1 = [88, 85, 90, 92, 91]
group2 = [84, 82, 88, 89, 86]

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(group1, group2)
print(f"Statistic: {stat}, p-value: {p_value}")
```

---

### 3.3 Kruskal-Wallis Test
- **Purpose**: Tests whether medians of three or more groups are equal.
- **Hypotheses**:
	- \(H_0\): All group medians are equal.
	- \(H_a\): At least one group median is different.

**Mathematical Formula**:

\[
H = \frac{12}{n(n+1)} \sum \frac{R_i^2}{n_i} - 3(n+1)
\]

**Python Example**:
```python
from scipy.stats import kruskal

# Three groups
group1 = [88, 85, 90, 92, 91]
group2 = [84, 82, 88, 89, 86]
group3 = [80, 81, 85, 87, 83]

# Perform Kruskal-Wallis test
stat, p_value = kruskal(group1, group2, group3)
print(f"Statistic: {stat}, p-value: {p_value}")
```

---

## 4. Chi-Square Tests

### 4.1 Chi-Square Test of Independence
- **Purpose**: Determines if two categorical variables are independent.
- **Hypotheses**:
	- \(H_0\): Variables are independent.
	- \(H_a\): Variables are not independent.

**Python Example**:
```python
data = np.array([[50, 30], [20, 100]])
chi2, p_value, dof, expected = chi2_contingency(data)
print(f"Chi2-statistic: {chi2}, p-value: {p_value}")
```

---

### 4.2 Chi-Square Goodness-of-Fit Test
- **Purpose**: Tests if an observed distribution matches an expected distribution.
- **Hypotheses**:
	- \(H_0\): Observed distribution matches the expected.
	- \(H_a\): Observed distribution does not match the expected.

**Python Example**:
```python
from scipy.stats import chisquare

# Observed and expected frequencies
observed = [18, 22, 20, 25, 15]
expected = [20, 20, 20, 20, 20]

# Perform goodness-of-fit test
chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
print(f"Chi2-statistic: {chi2}, p-value: {p_value}")
```

---

## Summary

This guide provides a detailed theoretical and practical understanding of hypothesis testing, including parametric and non-parametric methods. Each method includes assumptions, formulas, and Python code examples for real-world applications.