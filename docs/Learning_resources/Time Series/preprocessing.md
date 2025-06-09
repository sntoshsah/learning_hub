# ⏳ Time Series Preprocessing: Complete Guide

Time series preprocessing is a crucial step in time series analysis and modeling. Proper preparation improves model accuracy, stability, and interpretability. This guide covers the essential and advanced preprocessing techniques.

---

## 1. 📅 Handling Missing Timestamps

Time series data should have consistent and continuous timestamps. If timestamps are missing, you should reindex the series to a complete timeline and handle the missing values.

### 🔧 Code Example:

```python
# Reindexing time series to complete monthly frequency
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
df = df.reindex(full_index)
df.index.name = 'Month'
```

### 🧠 Handling missing values:

```python
# Fill missing with forward fill or interpolation
df['Passengers'].fillna(method='ffill', inplace=True)
# Or use interpolation
df['Passengers'].interpolate(method='linear', inplace=True)
```

---

## 2. 🔄 Resampling (Up-sampling & Down-sampling)

Resampling changes the frequency of the time series.

* **Down-sampling**: Reduce frequency (e.g., daily → monthly)
* **Up-sampling**: Increase frequency (e.g., monthly → daily)

### ✅ Code Example:

```python
# Down-sample to yearly
df_yearly = df['Passengers'].resample('Y').mean()

# Up-sample to daily with forward fill
df_daily = df['Passengers'].resample('D').ffill()
```

---

## 3. ⏪ Lag Features

Lag features help capture the relationship of past values with the current observation.

### ✅ Code Example:

```python
# Create lag features
for lag in range(1, 4):
    df[f'lag_{lag}'] = df['Passengers'].shift(lag)
```

---

## 4. 📉 Rolling Features

Rolling statistics (mean, std) are useful to capture short-term trends.

### ✅ Code Example:

```python
# Rolling mean and std
window = 12
df['roll_mean'] = df['Passengers'].rolling(window=window).mean()
df['roll_std'] = df['Passengers'].rolling(window=window).std()
```

---

## 5. 🔄 Differencing

Used to remove trend or seasonality to make a series stationary.

### ✅ Code Example:

```python
# First difference (to remove trend)
df['diff1'] = df['Passengers'].diff()

# Seasonal difference (to remove seasonality)
df['diff_seasonal'] = df['Passengers'].diff(12)
```

---

## 6. 📏 Stationarity Check (ADF & KPSS)

### ✅ Augmented Dickey-Fuller (ADF) Test:

```python
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df['Passengers'].dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
```

### ✅ KPSS Test:

```python
from statsmodels.tsa.stattools import kpss

kpss_result = kpss(df['Passengers'].dropna(), regression='c')
print(f"KPSS Statistic: {kpss_result[0]:.4f}")
print(f"p-value: {kpss_result[1]:.4f}")
```

* **ADF**: Null = non-stationary → reject if p < 0.05
* **KPSS**: Null = stationary → reject if p < 0.05

Use both to confirm results.

---

## 7. 🔢 Scaling and Normalization (Optional)

Useful for models that are sensitive to scale (e.g., LSTMs, neural networks).

### ✅ Code Example:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['scaled'] = scaler.fit_transform(df[['Passengers']])
```

---

## 8. 🧼 Outlier Detection & Smoothing

### Techniques:

* Z-score, IQR method
* Rolling median filter
* Winsorization

### Example:

```python
from scipy.stats import zscore
z_scores = zscore(df['Passengers'].dropna())
outliers = df[abs(z_scores) > 3]
```

---

## 9. 🛠 Feature Engineering for Time Series

Creating new informative features can enhance model performance.

### ✅ Time-based Features:

```python
# Extracting time-based features
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
```

### ✅ Fourier Series Features (to capture seasonality):

```python
import numpy as np

def create_fourier_terms(df, period, order):
    for i in range(1, order + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * df.index.dayofyear / period)
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * df.index.dayofyear / period)
    return df

df = create_fourier_terms(df, period=365, order=3)
```

---

## ✅ Summary: Preprocessing Checklist

| Task                | Description                             |
| ------------------- | --------------------------------------- |
| Missing timestamps  | Ensure consistent time index            |
| Resampling          | Change data frequency                   |
| Lag features        | Include past values                     |
| Rolling stats       | Capture local trends                    |
| Differencing        | Remove trend or seasonality             |
| ADF/KPSS            | Test for stationarity                   |
| Scaling             | Normalize for ML models                 |
| Outlier handling    | Detect and smooth anomalies             |
| Feature Engineering | Add temporal and Fourier-based features |

---

Next Steps:

* Train/Test split by date
* Model fitting (ARIMA, Prophet, etc.)

Would you like a code notebook or Markdown export of this complete preprocessing pipeline?
