# Advanced Hypothesis Testing in Python

## Introduction to Two-Sample Tests

Two-sample hypothesis testing allows us to compare statistics across different groups within our data. Let's explore this concept through both theoretical understanding and practical applications.

### Business Context and Real-World Applications

Consider these practical scenarios where two-sample tests are valuable:

1. A/B Testing in E-commerce:
    - Comparing conversion rates between two website designs
    - Analyzing customer spending across different marketing campaigns
    - Evaluating user engagement metrics between mobile and desktop users

2. HR Analytics:
    - Comparing salaries between different departments
    - Analyzing performance metrics between remote and office workers
    - Evaluating training program effectiveness

Let's implement these concepts using Python.

## T-Tests Implementation

### Two-Sample Independent T-Test

```python
import pandas as pd
import numpy as np
from scipy.stats import t
import seaborn as sns
import matplotlib.pyplot as plt

def perform_two_sample_ttest(data, group_column, value_column, group1, group2, alpha=0.05):
    """
    Performs a two-sample t-test with detailed analysis and visualization.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing the groups and values
    group_column : str
        Name of the column containing group labels
    value_column : str
        Name of the column containing the values to compare
    group1, group2 : str
        Names of the groups to compare
    alpha : float
        Significance level
    """
    # Extract the two groups
    sample1 = data[data[group_column] == group1][value_column]
    sample2 = data[data[group_column] == group2][value_column]
    
    # Calculate basic statistics
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sample1.mean(), sample2.mean()
    var1, var2 = sample1.var(ddof=1), sample2.var(ddof=1)
    
    # Calculate t-statistic
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / pooled_se
    
    # Calculate degrees of freedom
    df = n1 + n2 - 2
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create violin plots
    sns.violinplot(data=data, x=group_column, y=value_column)
    plt.title(f'Distribution Comparison: {group1} vs {group2}')
    
    # Print results
    results = {
        'Group 1 Mean': mean1,
        'Group 2 Mean': mean2,
        'Mean Difference': mean1 - mean2,
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant': p_value < alpha
    }
    
    return results, plt.gcf()

# Example usage with Stack Overflow data
stack_overflow = pd.read_csv('stack_overflow_data.csv')

results, fig = perform_two_sample_ttest(
    stack_overflow,
    'age_first_code_cut',
    'converted_comp',
    'child',
    'adult',
    alpha=0.05
)
```

### Paired T-Test Implementation

Paired t-tests are used when we have matched pairs of observations. Here's a comprehensive implementation:

```python
def perform_paired_ttest(data, before_col, after_col, alpha=0.05):
    """
    Performs a paired t-test with visualization and detailed analysis.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Dataset containing before and after measurements
    before_col, after_col : str
        Column names for before and after measurements
    alpha : float
        Significance level
    """
    # Calculate differences
    differences = data[after_col] - data[before_col]
    
    # Basic statistics
    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)
    n = len(differences)
    
    # Calculate t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    
    # Degrees of freedom
    df = n - 1
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before-After plot
    ax1.scatter(data[before_col], data[after_col])
    min_val = min(data[before_col].min(), data[after_col].min())
    max_val = max(data[before_col].max(), data[after_col].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel('Before')
    ax1.set_ylabel('After')
    ax1.set_title('Before vs After Measurements')
    
    # Differences histogram
    sns.histplot(differences, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title('Distribution of Differences')
    
    results = {
        'Mean Difference': mean_diff,
        'Standard Deviation of Differences': std_diff,
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant': p_value < alpha,
        'Confidence Interval': t.interval(1-alpha, df, 
                                       loc=mean_diff, 
                                       scale=std_diff/np.sqrt(n))
    }
    
    return results, fig

# Example with Republican voting data
repub_votes = pd.read_csv('republican_votes.csv')

results, fig = perform_paired_ttest(
    repub_votes,
    'repub_percent_08',
    'repub_percent_12',
    alpha=0.05
)
```

## ANOVA Testing

ANOVA (Analysis of Variance) is used when comparing more than two groups. Here's a comprehensive implementation:

```python
def perform_anova_analysis(data, group_column, value_column, alpha=0.05):
    """
    Performs one-way ANOVA with visualization and post-hoc analysis.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Dataset containing groups and values
    group_column : str
        Column name containing group labels
    value_column : str
        Column name containing values to compare
    alpha : float
        Significance level
    """
    import pingouin as pg
    
    # Perform ANOVA
    anova_results = pg.anova(data=data,
                            dv=value_column,
                            between=group_column)
    
    # Perform pairwise t-tests with Bonferroni correction
    posthoc = pg.pairwise_tests(data=data,
                               dv=value_column,
                               between=group_column,
                               padjust='bonf')
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Box plot
    sns.boxplot(data=data, x=group_column, y=value_column)
    plt.xticks(rotation=45)
    plt.title('Distribution by Group')
    
    # Add statistical annotations
    if anova_results['p-unc'].iloc[0] < alpha:
        plt.text(0.02, 0.98, 'Significant differences detected',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                color='red')
    
    return {
        'anova_results': anova_results,
        'posthoc_tests': posthoc,
        'plot': plt.gcf()
    }

# Example with job satisfaction data
results = perform_anova_analysis(
    stack_overflow,
    'job_sat',
    'converted_comp',
    alpha=0.05
)
```

## Statistical Power Analysis

Understanding statistical power is crucial for designing effective hypothesis tests:

```python
def calculate_power_analysis(data1, data2, alpha=0.05, n_simulations=1000):
    """
    Performs power analysis through simulation.
    
    Parameters:
    -----------
    data1, data2 : array-like
        The two groups to compare
    alpha : float
        Significance level
    n_simulations : int
        Number of simulations to run
    """
    from scipy import stats
    
    # Calculate effect size
    effect_size = (np.mean(data1) - np.mean(data2)) / \
                 np.sqrt((np.var(data1) + np.var(data2)) / 2)
    
    # Simulate tests
    significant_tests = 0
    
    for _ in range(n_simulations):
        # Resample with replacement
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        
        # Perform t-test
        _, p_value = stats.ttest_ind(sample1, sample2)
        
        if p_value < alpha:
            significant_tests += 1
    
    power = significant_tests / n_simulations
    
    return {
        'effect_size': effect_size,
        'power': power,
        'n_simulations': n_simulations
    }
```

## Best Practices and Guidelines

1. **Choosing the Right Test**
    - Use paired t-tests when observations are naturally paired
    - Use independent t-tests when comparing unrelated groups
    - Use ANOVA when comparing more than two groups
    - Consider non-parametric alternatives when assumptions are violated

2. **Sample Size Considerations**
    - Larger sample sizes increase statistical power
    - Use power analysis to determine required sample size
    - Consider practical significance alongside statistical significance

3. **Assumptions Checking**
    - Normality of distributions
    - Homogeneity of variances
    - Independence of observations

4. **Multiple Testing**
    - Apply appropriate corrections (e.g., Bonferroni) when performing multiple tests
    - Consider family-wise error rate
    - Be cautious of data dredging

5. **Reporting Results**
    - Always report effect sizes alongside p-values
    - Include confidence intervals
    - Provide clear visualizations
    - Document all decisions and assumptions

## Real-World Applications

### E-commerce Example

```python
# Example: Analyzing customer spending between mobile and desktop users
def analyze_platform_spending(data):
    """
    Analyzes spending patterns between mobile and desktop users.
    """
    results, fig = perform_two_sample_ttest(
        data,
        'platform',
        'spending',
        'mobile',
        'desktop'
    )
    
    # Additional business metrics
    roi_mobile = calculate_roi(data[data['platform'] == 'mobile'])
    roi_desktop = calculate_roi(data[data['platform'] == 'desktop'])
    
    return {
        'statistical_results': results,
        'visualization': fig,
        'business_metrics': {
            'mobile_roi': roi_mobile,
            'desktop_roi': roi_desktop
        }
    }
```

### HR Analytics Example

```python
# Example: Analyzing salary differences across departments
def analyze_salary_equity(data):
    """
    Performs comprehensive salary analysis across departments.
    """
    # Perform ANOVA
    anova_results = perform_anova_analysis(
        data,
        'department',
        'salary'
    )
    
    # Additional equity metrics
    gender_analysis = analyze_gender_pay_gap(data)
    experience_analysis = analyze_experience_impact(data)
    
    return {
        'anova_results': anova_results,
        'equity_metrics': {
            'gender_analysis': gender_analysis,
            'experience_impact': experience_analysis
        }
    }
```

This documentation provides a comprehensive guide to performing various types of hypothesis tests in Python, complete with practical examples and business applications. The code is structured to be both educational and immediately useful in real-world scenarios.