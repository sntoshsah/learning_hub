# 📘 Classical Time Series Models

---

## 1. 🔁 AR (AutoRegressive) Model

### ✅ Concept:

The AR model predicts the value of a time series using its **past values**.

### 📐 Mathematical Formula:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t
$$

* $y_t$: current value
* $\phi_i$: AR coefficients
* $p$: order
* $\epsilon_t$: white noise

### 🧪 Python Code:

```python
from statsmodels.tsa.ar_model import AutoReg

# Fit AR model
model_ar = AutoReg(df['Passengers'], lags=12).fit()
print(model_ar.summary())

# Predict
pred_ar = model_ar.predict(start=len(df), end=len(df)+11, dynamic=False)
```

---

## 2. 📉 MA (Moving Average) Model

### ✅ Concept:

Uses past forecast **errors** to model the current value.

### 📐 Mathematical Formula:

$$
y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
$$

### 🧪 Python Code:

```python
from statsmodels.tsa.arima.model import ARIMA

# MA(q) is ARIMA(0,0,q)
model_ma = ARIMA(df['Passengers'], order=(0, 0, 2)).fit()
print(model_ma.summary())
```

---

## 3. 🔁📉 ARMA (AutoRegressive Moving Average)

### ✅ Concept:

Combines AR and MA components.

### 📐 Mathematical Formula:

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t
$$

### 🧪 Python Code:

```python
# ARMA(p,q) is ARIMA(p,0,q)
model_arma = ARIMA(df['Passengers'], order=(2, 0, 2)).fit()
print(model_arma.summary())
```

---

## 4. 🔁⏬ ARIMA (AutoRegressive Integrated Moving Average)

### ✅ Concept:

ARMA + differencing for non-stationary series.

### 📐 Formula:

$$
\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i \Delta^d y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t
$$

* $d$: differencing order

### 🧪 Python Code:

```python
# ARIMA(p,d,q) where d is the differencing order
model_arima = ARIMA(df['Passengers'], order=(2, 1, 2)).fit()
print(model_arima.summary())
```

---

## 5. 📆 SARIMA (Seasonal ARIMA)

### ✅ Concept:

ARIMA + seasonal component

### 📐 Formula:

ARIMA(p,d,q)(P,D,Q,s)

* (p,d,q): non-seasonal
* (P,D,Q): seasonal
* $s$: seasonal period (e.g., 12 for monthly)

### 🧪 Python Code:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p,d,q)(P,D,Q,s)
model_sarima = SARIMAX(df['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
print(model_sarima.summary())
```

---

## 6. 🟦 SARIMAX (SARIMA with Exogenous Variables)

### ✅ Concept:

SARIMA + external predictors (e.g., holidays, Fourier terms)

### 🧪 Python Code:

```python
# Exogenous features (e.g., Fourier terms)
exog = df[['sin_1', 'cos_1']]

model_sarimax = SARIMAX(df['Passengers'], exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
print(model_sarimax.summary())
```

---

## 7. 📉 Exponential Smoothing (ETS)

### ✅ Concept:

Smoothing methods that estimate trend and seasonality using exponential decay.

### 📐 Types:

* Simple: no trend/seasonality
* Holt: linear trend
* Holt-Winters: trend + seasonality

### 🧪 Python Code:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Holt-Winters (additive)
model_ets = ExponentialSmoothing(
    df['Passengers'], trend='add', seasonal='add', seasonal_periods=12
).fit()

print(model_ets.summary())
```

---

## 📊 Visualization of Predictions

```python
plt.figure(figsize=(10,5))
plt.plot(df['Passengers'], label='Original')
plt.plot(model_arima.predict(start=120, end=132), label='ARIMA Forecast')
plt.plot(model_ets.fittedvalues, label='ETS Fitted')
plt.legend()
plt.title('Forecast Comparison')
plt.show()
```

---

## ✅ Summary Table

| Model   | Handles Trend | Handles Seasonality | Needs Stationarity | Exogenous Variables |
| ------- | ------------- | ------------------- | ------------------ | ------------------- |
| AR      | ✅             | ❌                   | ✅                  | ❌                   |
| MA      | ❌             | ❌                   | ✅                  | ❌                   |
| ARMA    | ✅             | ❌                   | ✅                  | ❌                   |
| ARIMA   | ✅             | ❌                   | ❌ (d handles it)   | ❌                   |
| SARIMA  | ✅             | ✅                   | ❌                  | ❌                   |
| SARIMAX | ✅             | ✅                   | ❌                  | ✅                   |
| ETS     | ✅             | ✅                   | ❌                  | ❌                   |

