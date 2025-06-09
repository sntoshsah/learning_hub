# 🔥 Deep Learning for Time Series with PyTorch

* RNN
* LSTM
* GRU
* 1D CNN
* Transformer-based models

---

## 🔁 1. Recurrent Neural Network (RNN)

### 📘 Concept:

* RNNs maintain a hidden state $h_t$
* Suitable for simple sequential patterns

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

---

### ✅ PyTorch Implementation:

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```

---

## 🧠 2. Long Short-Term Memory (LSTM)

### 📘 Concept:

* LSTM handles long-term dependencies with gates:

  * Forget, Input, Output

$$
c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t
\quad ; \quad h_t = o_t \cdot \tanh(c_t)
$$

---

### ✅ PyTorch Implementation:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

---

## 🚪 3. Gated Recurrent Unit (GRU)

### 📘 Concept:

* Simplified LSTM with fewer gates:

  * Update & Reset gates

$$
h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
$$

---

### ✅ PyTorch Implementation:

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
```

---

## 🧱 4. 1D CNN for Time Series

### 📘 Concept:

* Use convolution over time
* Good for local patterns

---

### ✅ PyTorch Implementation:

```python
class CNN1DModel(nn.Module):
    def __init__(self, input_channels, output_size, kernel_size=3):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.pool(self.relu(self.conv1(x))).squeeze(-1)
        return self.fc(x)
```

---

## 🔁 5. Transformer-Based Models

### 📘 Concept:

* Use **self-attention** to model global dependencies
* Handles long sequences well

---

### ✅ Simplified Transformer in PyTorch:

```python
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
```

---

## 🧪 Training Loop (Generic)

```python
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
```

---

## 📊 Example Dataset: AirPassengers or Synthetic Sinusoid

```python
import numpy as np
import pandas as pd

t = np.linspace(0, 100, 1000)
data = np.sin(0.2 * t) + np.random.normal(0, 0.1, len(t))
df = pd.DataFrame({'value': data})
```

---

## 🧠 Summary Table

| Model       | Handles Long-Term | Lightweight | Interpretable | Global View |
| ----------- | ----------------- | ----------- | ------------- | ----------- |
| RNN         | ❌                 | ✅           | ✅             | ❌           |
| LSTM        | ✅                 | ❌           | ❌             | ❌           |
| GRU         | ✅                 | ✅           | ❌             | ❌           |
| 1D CNN      | ❌ (local only)    | ✅           | ✅             | ❌           |
| Transformer | ✅                 | ❌ (costly)  | ❌             | ✅           |

---
