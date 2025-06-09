# **Deep Learning for NLP: A Comprehensive Tutorial**

## Introduction to Deep Learning for NLP

Deep Learning has revolutionized NLP by enabling models to:

    - Learn complex patterns in text
    - Handle variable-length sequences
    - Capture contextual meaning better than traditional methods

## **Table of Contents**
1. **RNNs & LSTMs for Text Classification**
2. **Seq2Seq Models with Attention**
3. **Transformer Implementation**
4. **Fine-Tuning Pretrained Models (BERT)**

---

## **1. RNNs & LSTMs for Text Classification**

### **Concept**
RNNs process sequential data by maintaining hidden states between time steps. LSTMs improve upon RNNs with gating mechanisms to better capture long-term dependencies.

### **PyTorch Implementation**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Sample data
texts = ["I loved the movie", "The film was terrible", "It was okay"]
labels = [1, 0, 1]  # 1=positive, 0=negative

# Tokenization and vocabulary
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Numericalize texts
text_pipeline = lambda x: vocab(tokenizer(x))
input_ids = [text_pipeline(text) for text in texts]

# Padding
padded_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq) for seq in input_ids],
    batch_first=True,
    padding_value=0
)

# LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))

# Training setup
model = SentimentLSTM(len(vocab), 100, 256)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Convert data to tensors
X = padded_sequences
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Prediction
test_text = "This was fantastic"
test_seq = torch.tensor([vocab(tokenizer(test_text))]).unsqueeze(0)
print(model(test_seq))  # Output close to 1 = positive
```

---

## **2. Seq2Seq with Attention**

### **Concept**
Seq2Seq models use an encoder-decoder architecture with attention mechanisms to focus on relevant parts of the input sequence when generating outputs.

### **PyTorch Implementation**
```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample data (English to French)
pairs = [("hello", "bonjour"), ("thanks", "merci")]

# Character-level vocabularies
src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

for eng, fr in pairs:
    for char in eng:
        if char not in src_vocab:
            src_vocab[char] = len(src_vocab)
    for char in fr:
        if char not in tgt_vocab:
            tgt_vocab[char] = len(tgt_vocab)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

# Attention Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        query = hidden.squeeze(0).unsqueeze(1)
        attention_weights = F.softmax(
            self.attention(torch.cat((query.expand(-1, encoder_outputs.size(0), -1), 
                                   encoder_outputs.permute(1, 0, 2)), dim=1),
            dim=1
        )
        context = torch.bmm(attention_weights, encoder_outputs.permute(1, 0, 2))
        rnn_input = torch.cat((embedded, context.permute(1, 0, 2)), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden, attention_weights

# Training setup
encoder = Encoder(len(src_vocab), 256)
decoder = Decoder(len(tgt_vocab), 256)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

# Training loop (simplified)
for epoch in range(100):
    for src, tgt in pairs:
        # Convert to tensors
        src_tensor = torch.tensor([src_vocab[c] for c in src])
        tgt_tensor = torch.tensor([tgt_vocab[c] for c in tgt])
        
        # Forward pass
        encoder_outputs, hidden = encoder(src_tensor)
        decoder_output, _, _ = decoder(tgt_tensor[:-1], hidden, encoder_outputs)
        
        # Loss and backprop
        loss = F.cross_entropy(decoder_output, tgt_tensor[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## **3. Transformer Implementation**

### **Concept**
Transformers use self-attention mechanisms to process all words in parallel, making them more efficient than RNNs for long sequences.

### **PyTorch Implementation**
```python
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# Example usage
model = TransformerModel(vocab_size=10000)
src = torch.randint(0, 10000, (10, 32))  # (seq_len, batch_size)
output = model(src)
```

---

## **4. Fine-Tuning Pretrained Models (BERT)**

### **Concept**
BERT (Bidirectional Encoder Representations from Transformers) is a powerful pretrained model that can be fine-tuned for specific NLP tasks.

### **PyTorch Implementation**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pretrained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Sample data
texts = ["This movie was great!", "Terrible experience"]
labels = [1, 0]  # 1=positive, 0=negative

# Tokenize inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels).unsqueeze(0)

# Forward pass
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop (simplified)
for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Prediction
test_text = ["I enjoyed this film"]
test_inputs = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    logits = model(**test_inputs).logits
predicted_class = torch.argmax(logits).item()
print("Positive" if predicted_class == 1 else "Negative")
```

---

## **Conclusion**
- **RNNs/LSTMs**: Good starting point for sequence modeling
- **Seq2Seq with Attention**: Powerful for translation/summarization
- **Transformers**: State-of-the-art architecture (BERT, GPT)
- **Fine-Tuning**: Leverage pretrained models for specific tasks

These PyTorch implementations provide a solid foundation for building advanced NLP systems. For production use, consider using HuggingFace's `transformers` library for optimized implementations.