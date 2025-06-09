# 📊 Exploratory Time Series Analysis (TSA)

Exploratory TSA is the first step in understanding the structure, behavior, and characteristics of your time series data. It includes visualizations, statistical summaries, and decomposition.

---

## 1. 📈 Plotting Time Series Data

The first and most important step is to **visualize** the data.

### 🧠 Objective:

* Identify trend, seasonality, outliers, sudden shifts

### ✅ Code Example:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sample time series data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Plot the raw time series
df.plot(title='Monthly Airline Passengers', figsize=(10, 5))
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.show()
```

---

## 2. 📉 Rolling Statistics (Moving Average and Std)

Rolling statistics smooth the time series by computing averages over a moving window.

### 🧠 Objective:

* Denoise the series
* Visualize local trends and variation

### ✅ Code Example:

```python
# Calculate rolling mean and std
rolling_mean = df['Passengers'].rolling(window=12).mean()
rolling_std = df['Passengers'].rolling(window=12).std()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Passengers'], label='Original')
plt.plot(rolling_mean, label='12-Month Rolling Mean')
plt.plot(rolling_std, label='12-Month Rolling Std', linestyle='--')
plt.title('Rolling Mean & Standard Deviation')
plt.legend()
plt.show()
```

---

## 3. 🔁 ACF and PACF (Autocorrelation & Partial Autocorrelation)

### 🧠 Objective:

* Detect seasonality, lags, and model order (e.g., ARIMA p and q)

* **ACF** shows correlation with **all previous lags**

* **PACF** shows **direct correlation** with specific lags, removing intermediate effects

### ✅ Code Example:

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF
plot_acf(df['Passengers'], lags=40)
plt.title("Autocorrelation Function (ACF)")
plt.show()

# PACF
plot_pacf(df['Passengers'], lags=40, method='ywm')
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()
```

> ℹ️ If your series is non-stationary, apply `.diff()` before plotting ACF/PACF.

---

## 4. 🔍 Seasonal Decomposition (Additive vs Multiplicative)

Time series can be broken into **Trend + Seasonality + Residual** components.

### ✅ Additive Model:

$$
Y(t) = T(t) + S(t) + R(t)
$$

### ✅ Multiplicative Model:

$$
Y(t) = T(t) \times S(t) \times R(t)
$$

### 🧠 When to use which?

* Use **additive** if seasonal fluctuations remain constant over time
* Use **multiplicative** if seasonal fluctuations **increase proportionally** with trend

### ✅ Code Example:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Multiplicative Decomposition
result_mul = seasonal_decompose(df['Passengers'], model='multiplicative')
result_mul.plot()
plt.suptitle("Multiplicative Decomposition", fontsize=14)
plt.show()

# Additive Decomposition (may not fit well here)
result_add = seasonal_decompose(df['Passengers'], model='additive')
result_add.plot()
plt.suptitle("Additive Decomposition", fontsize=14)
plt.show()
```

---

## 📌 Summary Table

| Task             | Purpose                              |
| ---------------- | ------------------------------------ |
| Plot time series | Understand structure, spot patterns  |
| Rolling mean/std | Visualize local trend and variance   |
| ACF              | Detect seasonality, lag correlation  |
| PACF             | Determine AR order in ARIMA          |
| Decomposition    | Extract trend, seasonality, residual |

---



