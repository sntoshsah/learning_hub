Here is a **complete tutorial on Fourier Series** with detailed explanations, illustrations, and Python code examples — designed to help you understand how Fourier terms are used in **time series analysis and feature engineering**, especially for **seasonality modeling**.

---

# 🎼 Fourier Series for Time Series Analysis

---

## 📘 What is a Fourier Series?

The **Fourier Series** represents a function (especially periodic ones) as a **sum of sine and cosine functions**.

For a time series $y(t)$, the Fourier series approximation is:

$$
y(t) = \sum_{k=1}^{K} \left[ a_k \cos\left(\frac{2\pi k t}{T} \right) + b_k \sin\left(\frac{2\pi k t}{T} \right) \right]
$$

* $T$: period of the seasonal cycle (e.g., 12 for months, 365 for daily data)
* $K$: number of harmonics (Fourier order)
* $a_k, b_k$: coefficients for cosine and sine terms

---

## 📌 Why Use Fourier Terms?

* Models complex and smooth **seasonal patterns**
* Often used in **Prophet**, **ARIMA with exogenous variables**, and **neural nets**
* Captures both **short-term** and **long-term** cyclic behaviors

---

## 🧮 Fourier Series in Python

### 🧰 Function to Create Fourier Features

```python
import numpy as np
import pandas as pd

def create_fourier_terms(df, time_col, period, order):
    """
    Add Fourier terms to a time series DataFrame.
    
    :param df: DataFrame with a datetime index
    :param time_col: Name of the datetime index or column
    :param period: Seasonality period (e.g., 365 for yearly)
    :param order: Number of harmonics
    :return: DataFrame with Fourier terms
    """
    df = df.copy()
    t = np.arange(len(df))
    
    for k in range(1, order + 1):
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    
    return df
```

### 📈 Example: Simulate Seasonal Time Series

```python
import matplotlib.pyplot as plt

# Simulate a signal with yearly seasonality
t = np.linspace(0, 365, 365)
y = np.sin(2 * np.pi * t / 365) + 0.5 * np.sin(2 * np.pi * 2 * t / 365)

plt.figure(figsize=(10, 4))
plt.plot(t, y, label='Simulated Seasonal Signal')
plt.title('Fourier Series Approximation of Seasonality')
plt.xlabel('Day of Year')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📊 Use Case: Add Fourier Terms to Real Data

```python
# Example using airline passenger dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# Add Fourier terms for seasonal pattern (monthly data: T = 12)
df_fourier = create_fourier_terms(df, time_col='Month', period=12, order=2)

df_fourier.head()
```

---

## 🤖 Fourier Features with Time Series Models

You can use Fourier features as **exogenous variables (X)** in time series models like:

### 🧠 ARIMA / SARIMAX:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    df['Passengers'], 
    exog=df_fourier[['sin_1', 'cos_1', 'sin_2', 'cos_2']], 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 12)
)
results = model.fit()
print(results.summary())
```

### 🕵️ Prophet:

```python
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

df_prophet = df.reset_index().rename(columns={"Month": "ds", "Passengers": "y"})

m = Prophet()
m.add_seasonality(name='yearly', period=365.25, fourier_order=4)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=24, freq='MS')
forecast = m.predict(future)

m.plot(forecast)
```

---

## 📌 Tips on Choosing Period and Order

| Use Case                          | Period | Fourier Order |
| --------------------------------- | ------ | ------------- |
| Monthly data (yearly seasonality) | 12     | 2–6           |
| Daily data (yearly seasonality)   | 365    | 4–10          |
| Weekly data (annual seasonality)  | 52     | 2–8           |

Higher order allows capturing more complex cycles but may lead to overfitting.

---

## ✅ Summary

| Concept        | Purpose                                   |
| -------------- | ----------------------------------------- |
| Fourier Series | Decompose signal into sine/cosine         |
| Period         | Time span of one seasonal cycle           |
| Order          | Number of harmonics                       |
| Use cases      | ARIMA, Prophet, deep learning models      |
| Benefits       | Capture smooth and non-linear seasonality |

---

## 📁 Optional: Export Fourier Features

```python
df_fourier.to_csv("data_with_fourier_features.csv")
```
