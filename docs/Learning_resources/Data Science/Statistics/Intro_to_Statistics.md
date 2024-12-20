# Chapter 1: Introduction to Statistics in Python

## What is Statistics?

Statistics is the practice and study of collecting, analyzing, and summarizing data.

### Summary Statistic
**Definition**: A summary statistic is a fact or summary derived from data.

- Example: "50% of friends drive to work."

---

## What Can Statistics Do?

- **Examples of Questions Answered by Statistics**:
  - How likely is someone to purchase a product?
  - What percentage of people drive to work?
  - A/B tests: Which ad is more effective in increasing sales?

### Limitations of Statistics
- Example: Statistics can analyze viewership data for a TV show like *Game of Thrones*, but it can't determine the root cause of its popularity.

---

## Types of Statistics

### Descriptive Statistics
- **Purpose**: Summarize and describe data.
- Example: "50% of people take the bus."

### Inferential Statistics
- **Purpose**: Make inferences about a population based on sample data.
- Example: Estimating the percentage of people who drive to work.

---

## Types of Data

### Numeric (Quantitative)
1. **Continuous (Measured)**: Examples include airplane speed, time spent waiting in line.
2. **Discrete (Counted)**: Examples include the number of pets or packages shipped.

### Categorical (Qualitative)
1. **Nominal (Unordered)**: Examples include marital status or country of residence.
2. **Ordinal (Ordered)**: Examples include survey responses such as "Strongly disagree" to "Strongly agree."

---

## Why Does Data Type Matter?

### Summary Statistics Example
```python
import numpy as np
np.mean(car_speeds['speed_mph'])  # Calculates the average car speed
```

### Value Counts Example
```python
demographics['marriage_status'].value_counts()
# Output:
# single      188
# married     143
# divorced    124
```

Data type affects how statistics and visualizations are applied.

---

## Measures of Center

### Mean
The average value of a dataset:
```python
import numpy as np
np.mean(msleep['sleep_total'])  # Calculates the mean sleep time
```

### Median
The middle value in a sorted dataset:
```python
np.median(msleep['sleep_total'])  # Calculates the median sleep time
```

### Mode
The most frequent value in a dataset:
```python
import statistics
statistics.mode(msleep['vore'])  # Finds the most common dietary type
```

---

## Measures of Spread

### Variance
The average squared deviation from the mean:
1. Subtract the mean from each data point.
2. Square the deviations.
3. Sum the squared deviations and divide by `n-1` (sample variance):
   ```python
   np.var(msleep['sleep_total'], ddof=1)  # Variance with sample correction
   ```

### Standard Deviation
The square root of variance:
```python
np.std(msleep['sleep_total'], ddof=1)  # Calculates standard deviation
```

---

## Quantiles and Interquartile Range (IQR)

### Quantiles
Divide data into intervals:
```python
np.quantile(msleep['sleep_total'], [0, 0.25, 0.5, 0.75, 1])
# Output: [1.9, 7.85, 10.1, 13.75, 19.9]
```

### IQR
The range between the 25th and 75th percentiles:
```python
from scipy.stats import iqr
iqr(msleep['sleep_total'])  # Calculates interquartile range
```

---

## Outliers

Outliers are data points that are substantially different from others:

- Calculation:
  ```python
  lower_threshold = Q1 - 1.5 * IQR
  upper_threshold = Q3 + 1.5 * IQR
  ```

### Example: Detecting Outliers
```python
msleep[(msleep['bodywt'] < lower_threshold) | (msleep['bodywt'] > upper_threshold)]
```

---

This document provides a comprehensive introduction to statistics, focusing on Python implementations using libraries like NumPy, pandas, and Matplotlib. Each section includes both conceptual explanations and practical examples.