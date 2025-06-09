# **Complete NLP Tutorial: Introduction to NLP & Text Preprocessing**  

This tutorial covers **Natural Language Processing (NLP) fundamentals** and **text preprocessing techniques** with Python code examples.  

---

## **1. What is NLP?**  
**Natural Language Processing (NLP)** is a branch of AI that enables computers to understand, interpret, and generate human language.  

### **Key Applications**  
| Application | Description | Example |
|-------------|-------------|---------|
| **Chatbots** | AI conversational agents | ChatGPT, Google Bard |
| **Machine Translation** | Text translation between languages | Google Translate |
| **Sentiment Analysis** | Detecting emotions in text | Twitter sentiment analysis |
| **Named Entity Recognition (NER)** | Identifying names, places, dates | Extracting "Apple" as a company |
| **Text Summarization** | Condensing long documents | News article summarization |

---

## **2. Text Preprocessing**  
Raw text data must be cleaned and normalized before NLP tasks.  

### **Key Steps in Text Preprocessing**  
1. **Tokenization**  
2. **Stemming & Lemmatization**  
3. **Stopword Removal**  
4. **Regex Cleaning**  

---

## **3. Tokenization**  
Splitting text into words, sentences, or subwords.  

### **Methods**  
| Method | Library | Use Case |
|--------|---------|----------|
| **Word Tokenization** | `nltk.word_tokenize()` | Splitting sentences into words |
| **Sentence Tokenization** | `nltk.sent_tokenize()` | Splitting paragraphs into sentences |
| **Subword Tokenization** | Hugging Face `Tokenizer` | Handling rare words (e.g., "unhappiness" → "un", "happiness") |

### **Example: Word & Sentence Tokenization**  
```python
import nltk
nltk.download('punkt')

text = "NLP is amazing! It helps computers understand language."

# Word Tokenization
words = nltk.word_tokenize(text)
print("Word Tokens:", words)  
# Output: ['NLP', 'is', 'amazing', '!', 'It', 'helps', 'computers', 'understand', 'language', '.']

# Sentence Tokenization
sentences = nltk.sent_tokenize(text)
print("Sentence Tokens:", sentences)  
# Output: ['NLP is amazing!', 'It helps computers understand language.']
```

---

## **4. Stemming vs. Lemmatization**  
Both reduce words to their base form, but lemmatization is more accurate.  

| Method | Example (Input → Output) | Library |
|--------|--------------------------|---------|
| **Stemming** | "running" → "run" | `PorterStemmer`, `SnowballStemmer` |
| **Lemmatization** | "better" → "good" | `WordNetLemmatizer` (requires POS tag) |

### **Example: Stemming & Lemmatization**  
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

text = "running runs ran better"

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in text.split()]
print("Stemmed:", stemmed)  
# Output: ['run', 'run', 'ran', 'better']

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in text.split()]  # 'v' for verb
print("Lemmatized:", lemmatized)  
# Output: ['run', 'run', 'run', 'better']
```

---

## **5. Stopword Removal**  
Stopwords (e.g., "the", "is", "and") add noise and are often removed.  

### **Example: Removing Stopwords**  
```python
from nltk.corpus import stopwords
nltk.download('stopwords')

text = "This is an example sentence showing off stopword filtration."
tokens = word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
filtered = [word for word in tokens if word not in stop_words and word.isalpha()]

print("Filtered:", filtered)  
# Output: ['example', 'sentence', 'showing', 'stopword', 'filtration']
```

---

## **6. Regex Cleaning**  
Removing unwanted characters (URLs, emails, punctuation).  

### **Example: Cleaning Text with Regex**  
```python
import re

text = "Check out https://example.com! Email me at user@email.com."

# Remove URLs
cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

# Remove emails
cleaned = re.sub(r'\S+@\S+', '', cleaned)

# Remove punctuation
cleaned = re.sub(r'[^\w\s]', '', cleaned)

print("Cleaned Text:", cleaned)  
# Output: "Check out  Email me at "
```

---

## **7. Full Text Preprocessing Pipeline**  
Combining all steps for clean text:  

```python
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs, emails
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

text = "NLP is awesome! Check https://nlp.org for more info."
print("Processed:", preprocess_text(text))  
# Output: "nlp awesome check info"
```

---

## **8. Libraries Comparison**  
| Task | NLTK | spaCy | TextBlob |
|------|------|-------|----------|
| Tokenization | ✅ | ✅ (faster) | ✅ |
| Lemmatization | ✅ (needs POS) | ✅ (automatic POS) | ✅ |
| Stopwords | ✅ | ✅ | ✅ |
| Sentiment Analysis | ❌ | ❌ | ✅ |

### **Example: spaCy for Faster Processing**  
```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion."
doc = nlp(text)

# Extract tokens, lemmas, and POS tags
for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

---

## **Summary**  
✅ **NLP** enables machines to work with human language.  
✅ **Text Preprocessing** includes tokenization, lemmatization, stopword removal, and regex cleaning.  
✅ **NLTK** is great for learning, **spaCy** for production, and **TextBlob** for quick sentiment analysis.  

Next Steps:  
➡️ Try these techniques on real datasets (e.g., Twitter data).  
➡️ Explore **feature extraction** (TF-IDF, Word2Vec).  

Would you like a tutorial on **Feature Engineering for NLP** next? 🚀