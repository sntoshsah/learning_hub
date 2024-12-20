# Hypothesis Testing in Python

## Introduction to Hypothesis Testing

Hypothesis testing is a fundamental concept in statistics that helps us make decisions about populations based on sample data. Let's explore this concept through both theoretical understanding and practical implementation in Python.

### Understanding A/B Testing

A/B testing is a practical application of hypothesis testing commonly used in business decisions. Consider the real-world example of Electronic Arts (EA) and SimCity 5:

- EA wanted to increase pre-orders of their game
- They tested different advertising scenarios
- Users were split into control and treatment groups
- The results showed that the treatment group (no ad) got 43.4% more purchases than the control group (with ad)

This raises an important question: Was this result statistically significant, or just due to chance? This is where hypothesis testing comes in.

## Working with Sample Data

Let's start with loading and examining our data:

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load the Stack Overflow Developer Survey data
stack_overflow = pd.read_csv('stack_overflow_data.csv')

# Example of examining the data
print(stack_overflow.head())
```

### Bootstrapping for Hypothesis Testing

Bootstrapping is a powerful technique for generating sampling distributions. Here's how to implement it:

```python
def generate_bootstrap_distribution(data, column, statistic_func, n_bootstraps=5000):
    """
    Generate a bootstrap distribution for a given statistic.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset to sample from
    column : str
        The column name to calculate the statistic on
    statistic_func : function
        The function to calculate the statistic (e.g., np.mean)
    n_bootstraps : int
        Number of bootstrap samples to generate
    
    Returns:
    --------
    list
        Bootstrap distribution of the statistic
    """
    boot_distn = []
    for _ in range(n_bootstraps):
        # Resample with replacement
        sample = data.sample(frac=1, replace=True)
        # Calculate and store the statistic
        boot_distn.append(statistic_func(sample[column]))
    
    return boot_distn

# Example usage
so_boot_distn = generate_bootstrap_distribution(
    stack_overflow, 
    'converted_comp', 
    np.mean
)

# Visualize the bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(so_boot_distn, bins=50, edgecolor='black')
plt.title('Bootstrap Distribution of Mean Compensation')
plt.xlabel('Mean Compensation')
plt.ylabel('Frequency')
plt.show()
```

## Z-Scores and Hypothesis Testing

### Understanding Z-Scores

Z-scores are standardized values that tell us how many standard deviations an observation is from the mean. The formula is:


z = (sample statistic - hypothesized parameter value) / standard error

```python
def calculate_z_score(sample_stat, hypoth_value, std_error):
    """
    Calculate the z-score for a hypothesis test.
    """
    return (sample_stat - hypoth_value) / std_error

# Example: Testing mean compensation
mean_comp_samp = stack_overflow['converted_comp'].mean()
mean_comp_hyp = 110000
std_error = np.std(so_boot_distn, ddof=1)

z_score = calculate_z_score(mean_comp_samp, mean_comp_hyp, std_error)
print(f"Z-score: {z_score:.3f}")
```

## P-Values and Statistical Significance

### Understanding P-Values

P-values represent the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true.

```python
def calculate_p_value(z_score, alternative='two-sided'):
    """
    Calculate p-value for a given z-score.
    
    Parameters:
    -----------
    z_score : float
        The calculated z-score
    alternative : str
        Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    --------
    float
        The p-value
    """
    if alternative == 'two-sided':
        return 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative == 'greater':
        return 1 - norm.cdf(z_score)
    else:  # alternative == 'less'
        return norm.cdf(z_score)

# Example usage
p_value = calculate_p_value(z_score, alternative='greater')
print(f"P-value: {p_value:.4f}")
```

### Statistical Significance

A result is considered statistically significant if the p-value is less than the chosen significance level (α). 

Common significance levels are:

- α = 0.05 (5% significance level)
- α = 0.01 (1% significance level)
- α = 0.10 (10% significance level)

```python
def test_hypothesis(p_value, alpha=0.05):
    """
    Make a decision about the hypothesis test.
    
    Parameters:
    -----------
    p_value : float
        The calculated p-value
    alpha : float
        The significance level
    
    Returns:
    --------
    str
        The decision and interpretation
    """
    if p_value <= alpha:
        return (f"Reject the null hypothesis (p={p_value:.4f} ≤ α={alpha}). "
                "There is sufficient evidence to support the alternative hypothesis.")
    else:
        return (f"Fail to reject the null hypothesis (p={p_value:.4f} > α={alpha}). "
                "There is insufficient evidence to support the alternative hypothesis.")

# Example usage
alpha = 0.05
decision = test_hypothesis(p_value, alpha)
print(decision)
```

### Confidence Intervals

Confidence intervals provide a range of plausible values for the population parameter:

```python
def calculate_confidence_interval(boot_distn, confidence_level=0.95):
    """
    Calculate confidence interval from a bootstrap distribution.
    """
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    lower = np.quantile(boot_distn, lower_percentile)
    upper = np.quantile(boot_distn, upper_percentile)
    
    return lower, upper

# Example usage
lower, upper = calculate_confidence_interval(so_boot_distn)
print(f"95% Confidence Interval: ({lower:.2f}, {upper:.2f})")
```

## Types of Errors in Hypothesis Testing

There are two types of errors that can occur in hypothesis testing:

1. Type I Error (False Positive):
	- Rejecting H₀ when it's actually true
	- Probability = α (significance level)

2. Type II Error (False Negative):
	- Failing to reject H₀ when it's actually false
	- Probability = β (depends on sample size and effect size)

## Best Practices for Hypothesis Testing

1. Always state your hypotheses clearly before conducting the test
2. Choose your significance level (α) before collecting data
3. Consider the practical significance, not just statistical significance
4. Report confidence intervals along with p-values
5. Be aware of multiple testing problems
6. Use appropriate sample sizes

## Example: Complete Hypothesis Test

Let's put it all together with a complete example:

```python
def complete_hypothesis_test(data, column, hypothesis_value, 
                           alternative='two-sided', alpha=0.05):
    """
    Conduct a complete hypothesis test.
    """
    # Calculate sample statistic
    sample_stat = data[column].mean()
    
    # Generate bootstrap distribution
    boot_distn = generate_bootstrap_distribution(data, column, np.mean)
    
    # Calculate standard error
    std_error = np.std(boot_distn, ddof=1)
    
    # Calculate z-score
    z_score = calculate_z_score(sample_stat, hypothesis_value, std_error)
    
    # Calculate p-value
    p_value = calculate_p_value(z_score, alternative)
    
    # Calculate confidence interval
    ci_lower, ci_upper = calculate_confidence_interval(boot_distn)
    
    # Make decision
    decision = test_hypothesis(p_value, alpha)
    
    return {
        'sample_statistic': sample_stat,
        'z_score': z_score,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'decision': decision
    }

# Example usage
results = complete_hypothesis_test(
    stack_overflow,
    'converted_comp',
    110000,
    alternative='greater',
    alpha=0.05
)

# Print results in a formatted way
print("Hypothesis Test Results")
print("=====================")
print(f"Sample Statistic: {results['sample_statistic']:.2f}")
print(f"Z-score: {results['z_score']:.3f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"95% Confidence Interval: ({results['confidence_interval'][0]:.2f}, "
      f"{results['confidence_interval'][1]:.2f})")
print("\nDecision:")
print(results['decision'])
```

This comprehensive guide provides both the theoretical foundation and practical implementation of hypothesis testing in Python. The code examples are designed to be reusable and adaptable for different hypothesis testing scenarios.