# 📈 Exponential Smoothing (ETS) – Complete Guide

Exponential smoothing is a family of forecasting methods that assign **exponentially decreasing weights** to past observations. It's useful when you expect **recent data** to be more important than older data.

---

## 🧠 Why Use Exponential Smoothing?

* Works well with **short-term forecasts**
* Handles **trend** and **seasonality**
* Easy to interpret and implement

---

## 🧩 Types of Exponential Smoothing

| Type                      | Captures                    | Use Case                 |
| ------------------------- | --------------------------- | ------------------------ |
| **Simple**                | Level only                  | No trend or seasonality  |
| **Holt’s Linear**         | Level + Trend               | Trending data            |
| **Holt-Winters (Triple)** | Level + Trend + Seasonality | Seasonal & trending data |

---

## 1. 📦 Simple Exponential Smoothing (SES)

Used when the data has **no trend or seasonality**.

### 📐 Formula:

$$
\hat{y}_{t+1} = \alpha y_t + (1 - \alpha) \hat{y}_t
$$

* $\alpha$: smoothing factor (0 < α < 1)

### ✅ Python Code:

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(df['Passengers']).fit(smoothing_level=0.2)
forecast = model.forecast(12)

df['Passengers'].plot(label='Actual', figsize=(10, 4))
forecast.plot(label='Forecast', color='red')
plt.title('Simple Exponential Smoothing')
plt.legend()
plt.show()
```

---

## 2. 📈 Holt’s Linear Trend Method

Used when the data has a **trend** but no seasonality.

### 📐 Formula:

* **Level**: $\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})$
* **Trend**: $b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$
* **Forecast**: $\hat{y}_{t+h} = \ell_t + h b_t$

### ✅ Python Code:

```python
from statsmodels.tsa.holtwinters import Holt

model = Holt(df['Passengers']).fit(smoothing_level=0.8, smoothing_trend=0.2)
forecast = model.forecast(12)

df['Passengers'].plot(label='Actual', figsize=(10, 4))
forecast.plot(label='Forecast', color='green')
plt.title("Holt's Linear Trend Forecast")
plt.legend()
plt.show()
```

---

## 3. 🔁 Holt-Winters (Triple Exponential Smoothing)

Handles **trend and seasonality**. Supports **additive** and **multiplicative** seasonality.

### 🔧 Use when:

* Additive: Seasonality magnitude **does not change** over time.
* Multiplicative: Seasonality **scales** with trend (grows or shrinks).

### 📐 Equations (Additive form):

* Level: $\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$
* Trend: $b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$
* Seasonal: $s_t = \gamma (y_t - \ell_t) + (1 - \gamma) s_{t-m}$
* Forecast: $\hat{y}_{t+h} = \ell_t + h b_t + s_{t+h-m(k+1)}$

Where:

* $m$: seasonal period (e.g., 12 for monthly)
* $k$: integer part of $\frac{h-1}{m}$

### ✅ Python Code (Additive Seasonality):

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    df['Passengers'],
    seasonal='add',
    trend='add',
    seasonal_periods=12
).fit()

forecast = model.forecast(12)

df['Passengers'].plot(label='Actual', figsize=(10, 4))
forecast.plot(label='Forecast', color='orange')
plt.title("Holt-Winters Additive Forecast")
plt.legend()
plt.show()
```

---

## 🔍 Choosing the Right ETS Model

| Pattern                  | Model                   |
| ------------------------ | ----------------------- |
| No trend, no seasonality | Simple Exponential      |
| Trend only               | Holt’s Linear           |
| Trend + Seasonality      | Holt-Winters (Add/Mult) |

For automatic selection, you can use the `auto_arima` from `pmdarima` or `statsforecast` in production.

---

## 📊 Visualizing Components

Use the `.plot_components()` for decomposition in tools like Prophet or manually using `seasonal_decompose`.

---

## 🧠 Key Points

* ETS methods are **suitable for short-term forecasting**.
* Easy to interpret and **fast to train**.
* Great as **baseline models** before trying ARIMA or ML methods.

