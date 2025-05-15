import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk
import spacy
import warnings

# Suppress warnings (e.g., np.bool8 deprecation)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download NLTK tokenizer data
try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading punkt_tab: {e}")
    exit(1)

# Load spaCy model for Russian preprocessing (optional for diploma)
try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    print("Ошибка: Модель 'ru_core_news_sm' не найдена. Установите её с помощью: python -m spacy download ru_core_news_sm")
    exit(1)

# Function for Russian text preprocessing (optional)
def preprocess_text_russian(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load the Emotion dataset (English, for lab work)
print("Loading Emotion dataset...")
try:
    emotion_dataset = load_dataset("emotion")
    train_data = emotion_dataset['train']
    test_data = emotion_dataset['test']
    texts = train_data['text'] + test_data['text']
    labels = train_data['label'] + test_data['label']
    label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Optional: For Russian dataset (uncomment to use)
# import pandas as pd
# df = pd.read_csv("your_russian_dataset.csv")
# texts = df["text"].tolist()
# labels = df["label"].tolist()
# label_names = ["sadness", "joy", "anger", "fear", "surprise", "love"]
# texts = [preprocess_text_russian(text) for text in texts]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# --- Approach 1: TfidfVectorizer + Logistic Regression ---
print("\n=== Approach 1: TfidfVectorizer + Logistic Regression ===")

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # Use stop_words=None for Russian

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train logistic regression model
tfidf_model = LogisticRegression(max_iter=1000, random_state=42)
tfidf_model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

# Evaluate
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf, recall_tfidf, f1_tfidf, _ = precision_recall_fscore_support(y_test, y_pred_tfidf, average='weighted')

print(f"Accuracy: {accuracy_tfidf:.4f}")
print(f"Precision: {precision_tfidf:.4f}")
print(f"Recall: {recall_tfidf:.4f}")
print(f"F1-Score: {f1_tfidf:.4f}")

# --- Approach 2: Word2Vec + Logistic Regression ---
print("\n=== Approach 2: Word2Vec + Logistic Regression ===")

# Load pre-trained Word2Vec model
print("Loading pre-trained Word2Vec model...")
try:
    word2vec_model = api.load("word2vec-google-news-300")
except Exception as e:
    print(f"Error loading Word2Vec model: {e}")
    exit(1)

# Optional: For Russian Word2Vec (uncomment to use)
# from gensim.models import KeyedVectors
# word2vec_model = KeyedVectors.load_word2vec_format("path_to_russian_word2vec.bin", binary=True)

# Function to convert text to Word2Vec embedding
def text_to_vector(text, model):
    words = word_tokenize(text.lower())
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(300)  # Return zero vector if no valid words

# Convert texts to Word2Vec embeddings
X_train_w2v = np.array([text_to_vector(text, word2vec_model) for text in X_train])
X_test_w2v = np.array([text_to_vector(text, word2vec_model) for text in X_test])

# Train logistic regression model
w2v_model = LogisticRegression(max_iter=1000, random_state=42)
w2v_model.fit(X_train_w2v, y_train)

# Predict on test set
y_pred_w2v = w2v_model.predict(X_test_w2v)

# Evaluate
accuracy_w2v = accuracy_score(y_test, y_pred_w2v)
precision_w2v, recall_w2v, f1_w2v, _ = precision_recall_fscore_support(y_test, y_pred_w2v, average='weighted')

print(f"Accuracy: {accuracy_w2v:.4f}")
print(f"Precision: {precision_w2v:.4f}")
print(f"Recall: {recall_w2v:.4f}")
print(f"F1-Score: {f1_w2v:.4f}")

# --- Comparison ---
print("\n=== Model Comparison ===")
print("TfidfVectorizer + Logistic Regression:")
print(f"Accuracy: {accuracy_tfidf:.4f}, Precision: {precision_tfidf:.4f}, Recall: {recall_tfidf:.4f}, F1-Score: {f1_tfidf:.4f}")
print("Word2Vec + Logistic Regression:")
print(f"Accuracy: {accuracy_w2v:.4f}, Precision: {precision_w2v:.4f}, Recall: {recall_w2v:.4f}, F1-Score: {f1_w2v:.4f}")

# Save results to CSV
results = pd.DataFrame({
    'Model': ['TfidfVectorizer', 'Word2Vec'],
    'Accuracy': [accuracy_tfidf, accuracy_w2v],
    'Precision': [precision_tfidf, precision_w2v],
    'Recall': [recall_tfidf, recall_w2v],
    'F1-Score': [f1_tfidf, f1_w2v]
})
results.to_csv("emotion_classification_results.csv", index=False)
print("\nResults saved to 'emotion_classification_results.csv'")