# 📘 Machine Learning for Time Series Forecasting

---

## ✅ 1. Why Use ML for Time Series?

Unlike statistical models (e.g., ARIMA), machine learning:

* Doesn’t assume linearity or stationarity
* Captures non-linear patterns and interactions
* Scales better for multivariate problems

---

## ✅ 2. Train/Test Split in Time Series

### ❌ Random split is invalid for time series!

* Time series data is **sequential**
* You must preserve **temporal order**

### ✅ Solution: Time-based split or **Walk-Forward Validation**

### 🔁 Walk-Forward Validation Process:

1. Train on `T1`, predict `T2`
2. Update training set with `T2`, predict `T3`
3. Repeat...

### ✅ Code Example:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

---

## ✅ 3. Feature Engineering for Time Series

### 📊 Key features:

* **Lag features**: past values (e.g., `y(t-1)`, `y(t-2)`)
* **Rolling stats**: rolling mean, std
* **Date/time**: month, day, hour, weekday, etc.

### ✅ Lag Features Example:

```python
for lag in range(1, 4):
    df[f'lag_{lag}'] = df['value'].shift(lag)
```

### ✅ Rolling Stats Example:

```python
df['rolling_mean'] = df['value'].rolling(3).mean()
df['rolling_std'] = df['value'].rolling(3).std()
```

### ✅ Time Features:

```python
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
```

---

## ✅ 4. ML Models for Forecasting

Let’s build a **regression model** to predict the next time point.

### ⚙️ Target:

$$
\hat{y}_t = f(y_{t-1}, y_{t-2}, \ldots, \text{date features})
$$

---

### 🔢 a. Linear Regression

#### ✅ Code:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
```

#### 📈 Pros:

* Fast and interpretable
* Performs well on linear trends

---

### 🌲 b. Random Forest

A tree-based ensemble model that captures non-linear patterns.

#### ✅ Code:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### ✅ Tips:

* Handles non-linearities and interactions
* Feature importance is interpretable

---

### ⚡ c. XGBoost

Gradient Boosted Trees: powerful & state-of-the-art

#### ✅ Code:

```python
import xgboost as xgb

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### ✅ Advantages:

* Handles missing values
* Fast and accurate
* Scalable

---

## 🔁 5. Multi-Step Forecasting

### 🧩 Two Approaches:

1. **Recursive Forecasting**

   * Predict next step
   * Feed it back into model to predict next

2. **Direct Forecasting**

   * Train one model per step (e.g., t+1, t+2)

### ✅ Recursive Example:

```python
def recursive_forecast(model, history, steps=5):
    preds = []
    for _ in range(steps):
        input_data = history[-n_lags:].reshape(1, -1)
        pred = model.predict(input_data)[0]
        preds.append(pred)
        history = np.append(history, pred)
    return preds
```

---

## ✅ 6. Evaluation Metrics

Use these to assess accuracy:

* **RMSE (Root Mean Squared Error)**:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

* **MAE (Mean Absolute Error)**:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

* **MAPE (Mean Absolute Percentage Error)**:

$$
MAPE = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

---

## 🧠 Additional Tips

* Scale your features (e.g., with `MinMaxScaler` or `StandardScaler`)
* Always validate on future (not random) data
* Start with linear, move to trees, test neural nets later

---

## 📚 Tools

* `scikit-learn`: regression models, pipelines
* `xgboost`: fast boosted trees
* `lightgbm`: efficient for large datasets
* `optuna`: hyperparameter optimization

---

## ✅ Final Thoughts

| ML Aspect               | Classical Time Series |
| ----------------------- | --------------------- |
| Lag-based features      | ✅                     |
| Handles multivariate    | ✅                     |
| Assumes stationarity    | ❌                     |
| Can model non-linearity | ✅                     |

---
