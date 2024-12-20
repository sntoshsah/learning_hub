# Chapter 3: Data Distribution

## Normal Distribution

### Overview
The normal distribution, also known as the Gaussian distribution, is characterized by:

- Symmetrical bell-shaped curve
- Total area under the curve equals 1
- The curve never reaches zero
- Defined by two parameters: mean (μ) and standard deviation (σ)

### Key Properties
1. **Standard Normal Distribution**
	- Mean = 0
	- Standard deviation = 1

2. **Areas under the Normal Distribution**
    - 68% of data falls within 1 standard deviation
    - 95% of data falls within 2 standard deviations
    - 99.7% of data falls within 3 standard deviations

### Working with Normal Distributions in Python

```python
from scipy.stats import norm

# Example using women's heights
# Mean = 161 cm, Standard deviation = 7 cm

# Calculate probability of being shorter than 154 cm
prob_shorter = norm.cdf(154, 161, 7)  # Returns 0.158655 (about 16%)

# Calculate probability of being taller than 154 cm
prob_taller = 1 - norm.cdf(154, 161, 7)  # Returns 0.841345 (about 84%)

# Calculate probability of height between 154-157 cm
prob_between = norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)  # Returns 0.1252

# Find height threshold where 90% of women are shorter
height_90th = norm.ppf(0.9, 161, 7)  # Returns 169.97086

# Generate random heights
random_heights = norm.rvs(161, 7, size=10)
```

## Central Limit Theorem (CLT)

### Overview
The Central Limit Theorem states that the sampling distribution of a statistic becomes closer to the normal distribution as the number of trials increases.

### Requirements
- Samples should be random and independent

### Implementation Example

```python
import pandas as pd
import numpy as np

# Create a die
die = pd.Series([1, 2, 3, 4, 5, 6])

# Function to generate sample means
def generate_sample_means(n_samples, sample_size):
    sample_means = []
    for i in range(n_samples):
        sample_means.append(np.mean(die.sample(sample_size, replace=True)))
    return sample_means

# Generate different numbers of sample means
sample_means_100 = generate_sample_means(100, 5)
sample_means_1000 = generate_sample_means(1000, 5)
```

## Poisson Distribution

### Overview
The Poisson distribution models the probability of events occurring over a fixed period when these events appear to happen at a certain rate but completely at random.

### Key Concepts
- Lambda (λ) represents the average number of events per time interval
- The distribution peaks at lambda
- Applicable to various scenarios like:
  - Animal shelter adoptions
  - Restaurant customer arrivals
  - Earthquake occurrences

### Python Implementation

```python
from scipy.stats import poisson

# Example: Average adoptions per week = 8
lambda_param = 8

# Probability of exactly 5 adoptions
prob_exact = poisson.pmf(5, lambda_param)  # Returns 0.09160366

# Probability of 5 or fewer adoptions
prob_less_equal = poisson.cdf(5, lambda_param)  # Returns 0.1912361

# Probability of more than 5 adoptions
prob_greater = 1 - poisson.cdf(5, lambda_param)  # Returns 0.8087639

# Generate random samples
random_samples = poisson.rvs(lambda_param, size=10)
```

## Exponential Distribution

### Overview
The exponential distribution models the probability of time between Poisson events.

### Key Properties
- Uses the same lambda (rate) as the Poisson distribution
- Continuous distribution (time)
- Expected value = 1/λ

### Python Implementation

```python
from scipy.stats import expon

# Example: Average 0.5 customer service tickets per minute
lambda_rate = 0.5
scale = 1/lambda_rate  # scale = 2

# Probability of waiting less than 1 minute
prob_less_1min = expon.cdf(1, scale=scale)

# Probability of waiting more than 4 minutes
prob_more_4min = 1 - expon.cdf(4, scale=scale)

# Probability of waiting between 1 and 4 minutes
prob_between = expon.cdf(4, scale=scale) - expon.cdf(1, scale=scale)
```

## Additional Distributions

### Student's t-Distribution
- Similar shape to normal distribution
- Has degrees of freedom (df) parameter
- Lower df = thicker tails
- Higher df = closer to normal distribution

### Log-Normal Distribution
- Variable whose logarithm is normally distributed
- Common applications:
  - Chess game lengths
  - Adult blood pressure
  - Hospital admissions during epidemics

## References
- Content developed by Maggie Matsui for DataCamp
- All code examples use SciPy's stats module
- Visualizations can be created using matplotlib or seaborn (not shown in examples)