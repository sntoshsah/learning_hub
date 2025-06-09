# ✅ Augmented Dickey-Fuller (ADF) Test

The **ADF Test** is a statistical test used to check whether a **time series is stationary**. It is one of the most commonly used methods in time series analysis.

---

## 📌 Why is Stationarity Important?

Most statistical models like **ARIMA**, **SARIMA**, and even **seasonal decomposition** assume that the **mean, variance, and autocorrelation** structure of the time series is constant over time (i.e., the series is stationary).

---

## 📐 What is the ADF Test?

The **Augmented Dickey-Fuller Test** tests the **null hypothesis**:

> H₀: The time series has a unit root (non-stationary)

> H₁: The time series is stationary

---

## 📊 ADF Regression Model:

ADF runs a regression of the following form:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \cdots + \delta_p \Delta y_{t-p} + \varepsilon_t
$$

Where:

* $\Delta y_t$ is the first difference of the series
* $\gamma$ is the coefficient tested against zero

If $\gamma < 0$ and **significantly different from zero**, the series is **stationary**.

---

## ✅ Interpreting ADF Test Results

* **ADF Statistic**: More **negative** → more likely to reject H₀
* **p-value**:

  * If **p < 0.05**: Reject H₀ → the series is **stationary**
  * If **p > 0.05**: Fail to reject H₀ → the series is **non-stationary**

---

## 🧪 Code Example: Run ADF Test

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Load data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# ADF Test
adf_result = adfuller(df['Passengers'])

# Output results
print("ADF Statistic: {:.4f}".format(adf_result[0]))
print("p-value: {:.4f}".format(adf_result[1]))
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")
```

---

## ⚠️ If the Series is Not Stationary

Use **differencing** or **log transformation**:

```python
# First-order differencing
df['diff'] = df['Passengers'].diff()

# Drop NA and re-test
adf_result_diff = adfuller(df['diff'].dropna())

print("\nAfter Differencing:")
print("ADF Statistic:", adf_result_diff[0])
print("p-value:", adf_result_diff[1])
```

---

## 📌 Summary Table

| Metric          | Description                                   |
| --------------- | --------------------------------------------- |
| ADF Statistic   | Test statistic for checking stationarity      |
| p-value         | If < 0.05 → data is stationary                |
| Critical Values | Thresholds at 1%, 5%, 10% levels              |
| Differencing    | Used to make non-stationary series stationary |

---

## 🧠 Bonus Tip: Visual Check of Stationarity

```python
import matplotlib.pyplot as plt

# Visual check
df['diff'].dropna().plot(title='First Differenced Series')
plt.grid(True)
plt.show()
```

