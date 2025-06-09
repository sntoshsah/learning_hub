# **Complete Guide to Intermediate NLP: Feature Engineering**

This tutorial covers essential feature engineering techniques for NLP, including Bag-of-Words (BoW), TF-IDF, and word embeddings (Word2Vec, GloVe, FastText) with Python implementations.

---

## **1. Bag-of-Words (BoW)**
### **What is BoW?**
- Represents text as a vector of word counts
- Ignores word order but captures frequency
- Creates a vocabulary from all unique words in corpus

### **Implementation with Scikit-learn**
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())
```

**Output:**
```
Vocabulary: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
BoW Matrix:
 [[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
```

### **Key Parameters**
- `max_features`: Limit vocabulary size
- `ngram_range`: Include word combinations (e.g., (1,2) for unigrams+bigrams)
- `stop_words`: Remove common words

---

## **2. TF-IDF (Term Frequency-Inverse Document Frequency)**
### **What is TF-IDF?**
- Measures word importance in a document relative to corpus
- Formula: `TF-IDF = TF(t,d) × IDF(t)`
  - TF: Term frequency in document
  - IDF: log(total docs / docs containing term)

### **Implementation**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

print("TF-IDF Matrix:\n", X_tfidf.toarray().round(2))
```

**Output:**
```
TF-IDF Matrix:
 [[0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]
 [0.   0.69 0.   0.28 0.   0.54 0.28 0.   0.28]
 [0.51 0.   0.   0.27 0.51 0.   0.27 0.51 0.27]
 [0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]]
```

### **When to Use TF-IDF vs BoW?**
- TF-IDF: When word importance matters (search, recommendations)
- BoW: When simple frequency is sufficient (quick prototypes)

---

## **3. Word Embeddings**
### **Why Word Embeddings?**
- BoW/TF-IDF lose semantic meaning
- Embeddings capture word relationships in dense vectors
- Similar words have similar vector representations

### **Types of Word Embeddings**
| Model | Description | Key Feature |
|-------|-------------|-------------|
| **Word2Vec** | Predicts words from context (CBOW) or context from words (Skip-gram) | Efficient, good for large corpora |
| **GloVe** | Uses global word co-occurrence statistics | Captures global patterns |
| **FastText** | Extends Word2Vec with subword information | Handles rare/unknown words |

---

## **4. Word2Vec Implementation**
### **Training Your Own Model**
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

sentences = [
    "Natural language processing is fascinating.",
    "Word embeddings capture semantic meaning.",
    "Deep learning models learn word representations."
]

# Tokenize sentences
tokenized = [word_tokenize(sent.lower()) for sent in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized,
    vector_size=100,  # Dimensionality of embeddings
    window=5,        # Context window size
    min_count=1,     # Ignore words with freq < this
    sg=1             # 1 for skip-gram, 0 for CBOW
)

# Get word vector
print("Vector for 'fascinating':", model.wv['fascinating'][:5])  # First 5 dimensions

# Find similar words
print("Similar to 'language':", model.wv.most_similar('language', topn=3))
```

### **Using Pre-trained Vectors**
```python
import gensim.downloader as api

# Download pre-trained model
w2v_model = api.load('word2vec-google-news-300')

# Example operations
print("Similar to 'king':", w2v_model.most_similar('king', topn=3))
print("Vector math: king - man + woman ≈", w2v_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
```

---

## **5. GloVe Implementation**
```python
# Download pre-trained GloVe
glove_model = api.load("glove-wiki-gigaword-300")

# Similar words
print("Similar to 'Paris':", glove_model.most_similar('Paris', topn=3))
```

---

## **6. FastText Implementation**
```python
# Download pre-trained FastText
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Handles out-of-vocabulary words via subwords
print("Vector for 'unhappiness':", fasttext_model['unhappiness'][:5])
print("Similar to 'sadness':", fasttext_model.most_similar('sadness', topn=3))
```

---

## **7. Choosing the Right Embedding**
| Scenario | Recommended Embedding |
|----------|-----------------------|
| Small dataset | Pre-trained Word2Vec/GloVe |
| Many rare words | FastText |
| Domain-specific text | Train your own Word2Vec |
| Multilingual tasks | FastText |

---

## **8. Practical Tips**
1. **Dimensionality**: Typically 100-300 dimensions
2. **Context Window**: 
   - Smaller (2-5): Syntactic tasks
   - Larger (10+): Semantic tasks
3. **Preprocessing**: Always clean text before training embeddings
4. **Evaluation**: Use analogies test set (e.g., `king - man + woman ≈ queen`)

---

## **Next Steps**
➡️ Try these techniques on your own text data  
➡️ Explore **document embeddings** (Doc2Vec, Sentence-BERT)  
➡️ Combine with machine learning models (LSTMs, Transformers)  
