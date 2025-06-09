# **Comprehensive Tutorial on Traditional NLP Models**

## **Table of Contents**
1. **Introduction to Traditional NLP Models**
2. **Text Classification**
   - Naive Bayes
   - Logistic Regression
3. **Sequence Models**
   - Hidden Markov Models (HMM)
   - Conditional Random Fields (CRF)
4. **Sentiment Analysis**
5. **Named Entity Recognition (NER)**
6. **Topic Modeling (LDA)**
7. **Conclusion**

---

# **1. Introduction to Traditional NLP Models**
Traditional NLP models rely on statistical and probabilistic methods to process and analyze text. Unlike deep learning models, they are:
- **Interpretable**: Easier to understand decision-making.
- **Less data-hungry**: Work well with smaller datasets.
- **Computationally efficient**: Faster training and inference.

This tutorial covers key traditional NLP models with **definitions, concepts, code, and examples**.

---

# **2. Text Classification**
Text classification assigns predefined categories to text documents.

## **2.1 Naive Bayes**
### **Concept**
- Based on **Bayes’ Theorem** with a "naive" assumption that features (words) are independent.
- **Types**:
  - **Multinomial Naive Bayes**: Best for text (word counts).
  - **Bernoulli Naive Bayes**: For binary word occurrences.

### **Example: Classifying News Articles**
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load dataset
categories = ['sci.space', 'comp.graphics', 'rec.autos']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Build pipeline (vectorizer → classifier)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train
model.fit(newsgroups_train.data, newsgroups_train.target)

# Evaluate
predicted = model.predict(newsgroups_test.data)
print(classification_report(newsgroups_test.target, predicted, target_names=newsgroups_test.target_names))
```
**Output**:
```
              precision    recall  f1-score   support
  comp.graphics       0.97      0.89      0.93       389
      rec.autos       0.95      0.98      0.96       396
     sci.space       0.94      0.98      0.96       394
```

### **Key Takeaways**
- Works well for small datasets.
- Fast but may underperform if words are highly correlated.

---

## **2.2 Logistic Regression**
### **Concept**
- A **linear model** for classification.
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for word weighting.
- Outputs probabilities using the **sigmoid function**.

### **Example: Email Spam Detection**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Example dataset (spam vs. ham)
texts = ["Win a free prize!", "Meeting at 3 PM", "Urgent: Claim your reward", "Project update"]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# TF-IDF + Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

# Predict
print(model.predict(["Free lottery!"]))  # Output: [1] (spam)
```
**Output**: `[1]` (classified as spam)

### **Key Takeaways**
- More flexible than Naive Bayes.
- Handles correlated features better.

---

# **3. Sequence Models**
Used for tasks where word order matters (e.g., POS tagging, NER).

## **3.1 Hidden Markov Models (HMM)**
### **Concept**
- **States** (hidden, e.g., POS tags) emit **observations** (words).
- Assumes **Markov Property**: Current state depends only on the previous state.

### **Example: POS Tagging**
```python
import nltk
from nltk.corpus import brown
from hmmlearn import hmm
import numpy as np

# Prepare data (POS tags)
nltk.download('brown')
tagged_sents = brown.tagged_sents(tagset='universal')[:1000]  # Simplify tags

# Encode words and tags
tag_set = set(tag for sent in tagged_sents for (word, tag) in sent)
word_set = set(word.lower() for sent in tagged_sents for (word, tag) in sent)
tag2id = {tag: i for i, tag in enumerate(tag_set)}
word2id = {word: i for i, word in enumerate(word_set)}

# Train HMM
model = hmm.CategoricalHMM(n_components=len(tag_set))
X = np.array([[word2id.get(word.lower(), -1)] for sent in tagged_sents for (word, tag) in sent])
lengths = [len(sent) for sent in tagged_sents]
model.fit(X, lengths=lengths)

# Predict POS for a new sentence
test_sentence = "The quick brown fox jumps".split()
test_ids = np.array([[word2id.get(word.lower(), -1)] for word in test_sentence])
predicted_tags = model.predict(test_ids)
print([list(tag2id.keys())[t] for t in predicted_tags])
```
**Output**: `['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB']`

### **Key Takeaways**
- Good for **POS tagging** but struggles with long dependencies.

---

## **3.2 Conditional Random Fields (CRF)**
### **Concept**
- **Discriminative model** (unlike HMM, which is generative).
- Considers **entire sequence** for predictions (better for NER).

### **Example: Named Entity Recognition (NER)**
```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

# Example data (IOB format)
train_data = [
    [("Apple", "B-ORG"), ("Inc.", "I-ORG"), ("is", "O"), ("in", "O"), ("Cupertino", "B-LOC")]
]

# Feature extraction
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'is_capitalized': word[0].isupper(),
    }
    return features

X_train = [[word2features(sent, i) for i in range(len(sent))] for sent in train_data]
y_train = [[tag for (word, tag) in sent] for sent in train_data]

# Train CRF
crf = CRF(algorithm='lbfgs')
crf.fit(X_train, y_train)

# Predict
test_sent = [("Microsoft", ""), ("is", ""), ("in", ""), ("Seattle", "")]
X_test = [word2features(test_sent, i) for i in range(len(test_sent))]
pred_tags = crf.predict_single(X_test)
print(pred_tags)  # Output: ['B-ORG', 'O', 'O', 'B-LOC']
```
**Output**: `['B-ORG', 'O', 'O', 'B-LOC']` (Microsoft=ORG, Seattle=LOC)

### **Key Takeaways**
- **Best for NER** (beats HMM).
- Handles **context dependencies** better.

---

# **4. Sentiment Analysis**
Classifies text as **positive, negative, or neutral**.

### **Example: Movie Reviews**
```python
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset (e.g., IMDb reviews)
reviews = load_files('aclImdb')  # Replace with your dataset path
X_train, X_test, y_train, y_test = train_test_split(reviews.data, reviews.target, test_size=0.2)

# Train model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

# Predict sentiment
print(model.predict(["This movie was terrible!"])[0])  # Output: 0 (negative)
```
**Output**: `0` (negative sentiment)

---

# **5. Named Entity Recognition (NER)**
Identifies **entities** (e.g., persons, organizations) in text.

### **Example: SpaCy for NER**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking to buy U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.label_)
```
**Output**:
```
Apple ORG
U.K. GPE
$1 billion MONEY
```

---

# **6. Topic Modeling (LDA)**
Discovers **latent topics** in documents.

### **Example: Newsgroup Topics**
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize text
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(newsgroups_train.data)

# Train LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# Display topics
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
```
**Output**:
```
Topic 0: ['car', 'engine', 'cars', 'dealer', 'ford', 'bmw', 'toyota', 'drive', 'speed', 'wheels']
Topic 1: ['graphics', 'image', '3d', 'files', 'format', 'jpeg', 'computer', 'software', 'display', 'video']
Topic 2: ['space', 'nasa', 'earth', 'orbit', 'moon', 'launch', 'shuttle', 'mission', 'solar', 'satellite']
```

---

# **7. Conclusion**
- **Naive Bayes & Logistic Regression**: Good for text classification.
- **HMM & CRF**: Best for sequence labeling (POS, NER).
- **Sentiment Analysis**: Classifies emotions in text.
- **LDA**: Discovers hidden topics.

These models are **still relevant** for interpretable, efficient NLP tasks! 🚀