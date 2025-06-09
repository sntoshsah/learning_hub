
# 📊 Advanced Time Series Topics Tutorial

---

## 1. 🔍 Anomaly Detection in Time Series

### ✅ Definition:

Detect points or periods where the behavior of the series deviates significantly from normal patterns.

### 📐 Methods:

* **Statistical Thresholding**:

  * $z = \frac{x - \mu}{\sigma}$
* **Rolling Statistics**:

  * Use rolling mean and standard deviation.
* **Model-based**:

  * Train a forecast model → compare actual vs predicted residuals.
* **Isolation Forest / One-Class SVM / Autoencoders**

### ✅ Code Example (Statistical):

```python
import numpy as np
threshold = 3  # Z-score
z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
anomalies = df[np.abs(z_scores) > threshold]
```

### ✅ Code Example (Autoencoder in PyTorch):

Let me know if you want the complete deep learning version.

---

## 2. 👯 Time Series Clustering

### ✅ Definition:

Grouping similar time series based on their shape, pattern, or frequency domain characteristics.

### 📐 Distance Metrics:

* **Euclidean**
* **Dynamic Time Warping (DTW)**: aligns sequences by warping time.

$$
DTW(x, y) = \min \sum \text{dist}(x_i, y_j) \text{ with time alignment}
$$

### ✅ Code Example (with `tslearn`):

```python
from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=3, metric="dtw")
labels = model.fit_predict(series_array)
```

---

## 3. 🔗 Multivariate Time Series

### ✅ Definition:

Multiple time-dependent variables recorded over the same time period.

### 📘 Vector Autoregression (VAR):

$$
Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + \dots + A_p Y_{t-p} + e_t
$$

### ✅ Code Example:

```python
from statsmodels.tsa.api import VAR

model = VAR(df[['temp', 'humidity']])
results = model.fit(maxlags=5)
forecast = results.forecast(df.values[-5:], steps=10)
```

---

## 4. 📎 Exogenous Variables

### ✅ Definition:

External variables that influence the target time series.

---

### 🔢 SARIMAX (ARIMA with exogenous):

$$
Y_t = ARIMA(p,d,q) + \beta X_t + \epsilon_t
$$

### ✅ Code Example:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['sales'], exog=df[['promo']], order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
```

---

### 🔢 VAR with Exogenous Variables:

```python
model = VAR(df[['temp', 'humidity']])
results = model.fit()
```

---

## 5. ❓ Forecasting with Missing Data

### ✅ Strategies:

* **Imputation** (mean, median, linear, interpolation)
* **Model-based Imputation** (Kalman, EM)
* **Multivariate Filling**

### ✅ Code Example:

```python
df['value'] = df['value'].interpolate(method='linear')
```

For more sophisticated:

```python
from fancyimpute import KNN
filled = KNN(k=3).fit_transform(df.values)
```

---

## 6. 📊 Probabilistic Forecasting

### ✅ Definition:

Forecasts that give uncertainty estimates (intervals or distributions), not just point estimates.

### 📘 Techniques:

* **Quantile Regression**: predict multiple quantiles
* **Bayesian models**: e.g., Pyro, TensorFlow Probability
* **Prophet**: gives prediction intervals
* **DeepAR (Amazon)**: RNN-based probabilistic model

### ✅ Quantile Loss:

$$
L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))
$$

### ✅ Example using Prophet:

```python
from prophet import Prophet
model = Prophet()
model.fit(df.rename(columns={"value": "y", "date": "ds"}))
forecast = model.predict(model.make_future_dataframe(periods=30))
```

