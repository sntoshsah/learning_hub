# **Transformer Models & BERT: A Comprehensive PyTorch Tutorial**

This tutorial provides an in-depth exploration of Transformer models, covering attention mechanisms, BERT/GPT/T5 architectures, and fine-tuning pre-trained models using PyTorch and Hugging Face.

## **Table of Contents**
1. **Attention Mechanism & Self-Attention**
2. **Transformer Architecture**
3. **BERT Architecture**
4. **GPT Architecture**
5. **T5 Architecture**
6. **Fine-tuning Pre-trained Models with Hugging Face**
7. **Conclusion**

---

## **1. Attention Mechanism & Self-Attention**

### **Concept**
- **Attention Mechanism**: Allows models to focus on relevant parts of input sequences when making predictions.
- **Self-Attention**: A variant where input sequences attend to themselves to capture relationships between all positions.

### **Key Components**
- **Query (Q)**: What the model is looking for
- **Key (K)**: What the input contains
- **Value (V)**: Actual content to be weighted

### **PyTorch Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Example usage
embed_size = 256
heads = 8
input_tensor = torch.randn(1, 10, embed_size)  # (batch, seq_len, embed_size)
self_attn = SelfAttention(embed_size, heads)
output = self_attn(input_tensor, input_tensor, input_tensor, mask=None)
print(output.shape)  # torch.Size([1, 10, 256])
```

---

## **2. Transformer Architecture**

### **Concept**
The Transformer model consists of:
- **Encoder**: Processes input sequence
- **Decoder**: Generates output sequence
- **Multi-Head Attention**: Multiple attention heads capture different relationships
- **Position-wise Feed Forward Networks**: Processes each position separately
- **Residual Connections & Layer Normalization**: Helps training deep networks

### **PyTorch Implementation**
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
```

---

## **3. BERT Architecture**

### **Concept**
- **Bidirectional Encoder Representations from Transformers**
- Uses masked language modeling (MLM) and next sentence prediction (NSP)
- Processes all words simultaneously (not sequential like RNNs)

### **PyTorch Implementation (Simplified)**
```python
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example usage
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pooler_output = outputs.pooler_output

print(f"Last hidden states shape: {last_hidden_states.shape}")  # [1, 7, 768]
print(f"Pooler output shape: {pooler_output.shape}")  # [1, 768]
```

---

## **4. GPT Architecture**

### **Concept**
- **Generative Pre-trained Transformer**
- Uses decoder-only transformer
- Autoregressive generation (predicts next token given previous ones)

### **PyTorch Implementation**
```python
from transformers import GPT2Model, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(f"Hidden states shape: {last_hidden_states.shape}")  # [1, 7, 768]
```

---

## **5. T5 Architecture**

### **Concept**
- **Text-to-Text Transfer Transformer**
- All NLP tasks framed as text-to-text problems
- Uses both encoder and decoder

### **PyTorch Implementation**
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

input_ids = tokenizer(
    "translate English to German: The house is wonderful", 
    return_tensors="pt"
).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "Das Haus ist wunderbar"
```

---

## **6. Fine-tuning Pre-trained Models with Hugging Face**

### **Concept**
Fine-tuning adapts pre-trained models to specific tasks by continuing training on domain-specific data.

### **Example: Text Classification with BERT**
```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Sample data
train_texts = ["Great movie!", "Terrible experience", "It was okay"]
train_labels = [1, 0, 1]  # 1=positive, 0=negative

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create dataset
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune
trainer.train()

# Prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs.argmax().item()

print(predict("I loved this film!"))  # Output: 1 (positive)
```

---

## **7. Conclusion**
- **Attention Mechanisms**: Enable models to focus on relevant input parts
- **Transformer Architectures**: Foundation for BERT, GPT, T5
- **Pre-trained Models**: Powerful starting points for NLP tasks
- **Fine-tuning**: Adapts models to specific domains with minimal data

This tutorial provides the essential knowledge and PyTorch implementations to work with modern Transformer models. For production use, leverage the Hugging Face ecosystem for optimized implementations.