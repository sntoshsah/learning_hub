# ARIMA: AutoRegressive Integrated Moving Average
# =========================================================
## 📌 What is ARIMA?

**ARIMA** stands for:

* **AR**: AutoRegressive (depends on past values)
* **I**: Integrated (uses differencing to make the series stationary)
* **MA**: Moving Average (depends on past forecast errors)

ARIMA models are used to **forecast non-stationary time series data** by first transforming it into a stationary series using differencing and then modeling the transformed data using ARMA.

---

## ✅ When to Use ARIMA?

Use ARIMA when:

* Your data shows a **trend** or **non-stationarity** (checked via ADF/KPSS tests)
* There's **no clear seasonality** (for that, use SARIMA)

---

## 📐 Mathematical Breakdown of ARIMA(p, d, q)

$$
\underbrace{(1 - \sum_{i=1}^p \phi_i L^i)}_{\text{AR(p)}} \cdot (1 - L)^d y_t = \underbrace{(1 + \sum_{j=1}^q \theta_j L^j)}_{\text{MA(q)}} \cdot \epsilon_t
$$

Where:

* $L$: Lag operator (e.g., $L y_t = y_{t-1}$)
* $p$: Number of AR terms
* $d$: Number of differences
* $q$: Number of MA terms
* $\phi_i$: AR coefficients
* $\theta_j$: MA coefficients
* $\epsilon_t$: White noise

---

## 🔁 Components

| Component | Description                                          | Python Equivalent |
| --------- | ---------------------------------------------------- | ----------------- |
| `p`       | # of past lags (autoregression)                      | `AR()`            |
| `d`       | # of times to difference the series to be stationary | `series.diff(d)`  |
| `q`       | # of past forecast errors (moving average)           | `MA()`            |

---

## 📉 Differencing (The “I” in ARIMA)

Differencing helps **remove trends** and **stabilize the mean**.

### 1st Order Differencing:

$$
y'_t = y_t - y_{t-1}
$$

### 2nd Order Differencing:

$$
y''_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})
$$

You only difference until the series becomes **stationary** (usually $d \in \{0, 1, 2\}$).

---

## 🧪 Full Python Example: ARIMA

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your time series data
df = pd.read_csv('air_passengers.csv', index_col='Month', parse_dates=True)
df = df.asfreq('MS')

# Visualize original series
df['Passengers'].plot(title='Original Time Series')
plt.show()

# Fit ARIMA(2,1,2): AR(2), 1st diff, MA(2)
model_arima = ARIMA(df['Passengers'], order=(2, 1, 2))
fitted_model = model_arima.fit()

# Summary of coefficients and diagnostics
print(fitted_model.summary())

# Forecast next 12 months
forecast = fitted_model.forecast(steps=12)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Passengers'], label='Original')
plt.plot(forecast.index, forecast, label='ARIMA Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecasting')
plt.show()
```

---

## 🔍 Model Interpretation

From `model.summary()` you'll get:

* Coefficients of AR and MA terms
* Standard errors and p-values
* AIC/BIC: Model selection criteria
* Residual diagnostics

---

## 🔍 Forecast vs Actual Example

You can split the data for **training and testing** and compare actual vs predicted:

```python
train = df['Passengers'][:120]
test = df['Passengers'][120:]

model_arima = ARIMA(train, order=(2, 1, 2)).fit()
preds = model_arima.forecast(steps=len(test))

plt.figure(figsize=(10, 5))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(preds.index, preds, label='Predicted', linestyle='--')
plt.legend()
plt.title('Train/Test Split with ARIMA Forecast')
plt.show()
```

---

## 🧠 Key Takeaways

* ARIMA models the **past values and forecast errors**.
* The `d` term (differencing) is crucial for **removing non-stationarity**.
* Proper preprocessing (e.g., ADF/KPSS tests, differencing) is **essential** before fitting ARIMA.
* Always **check residuals** to ensure the model captures all structure (should be white noise).

