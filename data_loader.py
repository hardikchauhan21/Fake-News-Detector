"""
Data loading and preprocessing utilities for Fake News Detector
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(fake_path, true_path):
    """
    Load fake and real news datasets

    Args:
        fake_path (str): Path to fake news CSV
        true_path (str): Path to real news CSV

    Returns:
        pd.DataFrame: Combined dataset with labels
    """
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df['label'] = 0  # 0 for fake
    true_df['label'] = 1  # 1 for real

    # Combine
    df = pd.concat([fake_df, true_df], axis=0)
    df = df.reset_index(drop=True)

    # Combine title and text for better context
    df['content'] = df['title'] + " " + df['text']

    return df

def clean_text(text):
    """
    Clean and preprocess text data

    Args:
        text (str): Raw text

    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def remove_stopwords(text):
    """
    Remove stopwords from text

    Args:
        text (str): Cleaned text

    Returns:
        str: Text without stopwords
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def stem_text(text):
    """
    Apply stemming to text

    Args:
        text (str): Text without stopwords

    Returns:
        str: Stemmed text
    """
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def preprocess_pipeline(text):
    """
    Full preprocessing pipeline

    Args:
        text (str): Raw text

    Returns:
        str: Preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

if __name__ == "__main__":
    # Example usage
    print("Data loader module for Fake News Detector")
    print("Usage: from src.data_loader import load_data, preprocess_pipeline")