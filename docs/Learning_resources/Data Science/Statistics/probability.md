# Chapter 2: Measuring Chance

## What Are the Chances?

### Probability Basics
- **Probability Definition**: The probability of an event is the measure of the likelihood that the event will occur. The formula for calculating probability is:

\[
P(\text{event}) = \frac{\text{# of favorable outcomes}}{\text{total number of possible outcomes}}
\]

  **Example**: 
  Consider the probability of flipping heads in a fair coin:

  - There are 2 possible outcomes: heads or tails.
  - The number of favorable outcomes (heads) is 1.
  Therefore:

\[
P(\text{heads}) = \frac{1}{2} = 50\%
\]

  **Code Example**:
  ```python
  # Simulating a coin flip
  import random

  outcomes = ['heads', 'tails']
  result = random.choice(outcomes)
  print(f"Coin flip result: {result}")
  ```

---

## Sampling

### Sampling from a DataFrame
- **Definition**: Sampling refers to randomly selecting rows from a DataFrame. In Python, you can use the `sample()` method from pandas to achieve this.

**Code Example**:
```python
import pandas as pd

# Sample data representing sales counts
sales_counts = pd.DataFrame({
    'name': ['Amir', 'Brian', 'Claire', 'Damian'],
    'n_sales': [178, 128, 75, 69]
})

# Randomly sample one row
sampled = sales_counts.sample()
print(sampled)
```

This will randomly select one row from the `sales_counts` DataFrame.

---

### Setting a Random Seed
- **Definition**: A random seed is used to initialize the random number generator so that the results are reproducible. By setting a seed, you ensure that the random numbers generated (e.g., when sampling data) are the same every time the code is run.

**Code Example**:
```python
import numpy as np

# Set a random seed for reproducibility
np.random.seed(10)

# Sample from the sales_counts DataFrame
sampled = sales_counts.sample()
print(sampled)
```

Setting the seed ensures that the same row is selected every time this code is executed.

---

## Replacement in Sampling

### Sampling Without Replacement
- **Definition**: Sampling without replacement means that once an item is selected, it is not returned to the population. This results in dependent events, as each selection affects the pool for subsequent selections.

**Code Example**:
```python
# Sampling without replacement
sampled_2 = sales_counts.sample(2, replace=False)
print(sampled_2)
```

This will sample two rows, but the same row cannot be selected twice.

### Sampling With Replacement
- **Definition**: Sampling with replacement means that each item selected is returned to the population, allowing the same item to be selected multiple times. This results in independent events, as each selection is independent of the others.

**Code Example**:
```python
# Sampling with replacement
sampled_3 = sales_counts.sample(5, replace=True)
print(sampled_3)
```

This will sample five rows, and rows can be repeated in the output.

---

## Discrete Probability Distributions

### Probability Distribution
- **Definition**: A probability distribution describes the probability of each possible outcome in a scenario. For example, when rolling a fair die, each face has an equal probability of landing.

**Example**: For a fair die, the probability distribution is uniform:

\[
P(\text{roll} = x) = \frac{1}{6}, \text{ where } x \in \{1, 2, 3, 4, 5, 6\}
\]

**Code Example**:
```python
import numpy as np

# Simulate 1000 rolls of a fair die
die_rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=1000)
print("Sample of 10 rolls:", die_rolls[:10])
```

This will simulate 1000 rolls of a fair die, with each number having an equal chance of being rolled.

### Expected Value
- **Definition**: The expected value (or mean) of a probability distribution is the average value that would be obtained from a large number of trials. The expected value for a fair die roll is calculated as:

\[
\text{Expected value} = \frac{1}{6} \times (1 + 2 + 3 + 4 + 5 + 6) = 3.5
\]

**Code Example**:
```python
# Calculate the expected value of a fair die roll
expected_value = np.mean([1, 2, 3, 4, 5, 6])
print(f"Expected value of a fair die roll: {expected_value}")
```

---

### Law of Large Numbers
- **Definition**: The Law of Large Numbers states that as the sample size increases, the sample mean will approach the expected value. This is particularly important when sampling from distributions.

**Code Example**:
```python
# Simulate a larger sample of die rolls
rolls_1000 = np.random.choice([1, 2, 3, 4, 5, 6], size=1000)
mean_1000 = np.mean(rolls_1000)
print(f"Mean of 1000 rolls: {mean_1000}")
```

The mean of a large sample of rolls will approach the expected value of 3.5.

---
