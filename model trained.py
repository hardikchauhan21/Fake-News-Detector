# %%
# Set the path to include the project root and change working directory
import sys
sys.path.append("/Users/hardikchauhan/fake-news-detector")
import os
os.chdir("/Users/hardikchauhan/fake-news-detector")

# %% [markdown]
# # Fake News Detector - EDA and Modeling
#
# This notebook explores the fake news dataset and builds a classification model.
#
# ## Steps:
# 1. Load and explore the dataset
# 2. Text preprocessing
# 3. Feature engineering (TF-IDF)
# 4. Model training and evaluation
# 5. Error analysis
# 6. Model explainability with LIME
# 7. Saving the model and vectorizer

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data, preprocess_pipeline
import re
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

# %% [markdown]
# # Load dataset
#
# NOTE: You need to download the dataset from Kaggle first:
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Place Fake.csv and True.csv in the data/ directory

# %%
# Uncomment and run after downloading data:
df = load_data('data/Fake.csv', 'data/True.csv')
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# # Sample Data Structure (for reference)
#
# Once you have the data, it will have columns like:
# - : News headline
# - : Article body
# - : News topic
# - Sat Sep  5 00:22:27 IST 2026: Publication date
# - : 0 for fake, 1 for real (added during loading)

# %%
# Basic info
print(df.info())
print(f"\nFake news count: {df['label'].value_counts()[0]}")
print(f"Real news count: {df['label'].value_counts()[1]}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# %% [markdown]
# ## Text length analysis
#
# We'll look at the length of the text and content (title + text).

# %%
# Text length analysis
df['text_length'] = df['text'].str.len()
df['content_length'] = df['content'].str.len()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='text_length', hue='label', bins=50, alpha=0.7)
plt.title('Text Length Distribution')
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='content_length', hue='label', bins=50, alpha=0.7)
plt.title('Content Length Distribution')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Text Preprocessing
#
# We'll apply our preprocessing pipeline to the content column.

# %%
# Apply preprocessing
df['cleaned_content'] = df['content'].apply(preprocess_pipeline)
df[['content', 'cleaned_content']].head()

# %% [markdown]
# ## Feature Engineering: TF-IDF
#
# Convert text to numerical features using TF-IDF vectorization.

# %%
# From sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Prepare features and target
X = df['cleaned_content']
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Try different TF-IDF configurations
tfidf_configs = [
    {"max_features": 5000, "ngram_range": (1, 1), "name": "Unigram_5k"},
    {"max_features": 5000, "ngram_range": (1, 2), "name": "Bigram_5k"},
    {"max_features": 10000, "ngram_range": (1, 2), "name": "Bigram_10k"},
    {"max_features": 5000, "ngram_range": (1, 3), "name": "Trigram_5k"}
]

# Store results
results = []

for config in tfidf_configs:
    print(f"\n{'='*50}")
    print(f"Testing TF-IDF config: {config['name']}")
    print(f"{'='*50}")

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=config['max_features'],
                           ngram_range=config['ngram_range'])
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"Features: {X_train_tfidf.shape[1]}")

    # Try different models
    models = [
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
        ("SVM", SVC(random_state=42, probability=True, class_weight='balanced')),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
    ]

    for model_name, model in models:
        # Train model
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        results.append({
            'tfidf_config': config['name'],
            'model': model_name,
            'accuracy': accuracy,
            'tfidf_object': tfidf,
            'model_object': model
        })

        print(f"{model_name}: Accuracy = {accuracy:.4f}")

# Find best performing model
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_result['model']} with {best_result['tfidf_config']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"{'='*60}")

# Use the best model and vectorizer for further analysis
best_tfidf = best_result['tfidf_object']
best_model = best_result['model_object']

# Transform data with best TF-IDF
X_train_best = best_tfidf.transform(X_train)
X_test_best = best_tfidf.transform(X_test)

# Get predictions from best model
y_pred_best = best_model.predict(X_test_best)
y_pred_proba_best = best_model.predict_proba(X_test_best)

print(f"\nBest model performance details:")
print(classification_report(y_test, y_pred_best, target_names=['Fake', 'Real']))

# %% [markdown]
# ## Model Training
#
# We'll compare Logistic Regression and Naive Bayes classifiers.

# %%
# Train Logistic Regression with balanced class weights to counter bias
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_tfidf)
y_pred_proba_lr = lr_model.predict_proba(X_test_tfidf)

# Evaluate
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Fake', 'Real']))

# %%
# Train Naive Bayes for comparison
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# Compare models
print(f"\nAccuracy Comparison:")
print(f"Logistic Regression: {lr_accuracy:.4f}")
print(f"Naive Bayes: {nb_accuracy:.4f}")

# %% [markdown]
# ## Confusion Matrix
#
# Let's visualize the performance of our best model.

# %%
# Plot confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Fake', 'Real'],
             yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## Error Analysis
#
# Let's look at some examples the model got wrong to understand its limitations.

# %%
# Find misclassified examples
misclassified_idx = y_test != y_pred_lr
X_test_misclassified = X_test[misclassified_idx]
y_test_misclassified = y_test[misclassified_idx]
y_pred_misclassified = y_pred_lr[misclassified_idx]

print(f"Number of misclassified samples: {len(X_test_misclassified)}")

# Show a few examples
for i in range(min(3, len(X_test_misclassified))):
    print(f"\nExample {i+1}:")
    print(f"True label: {'Real' if y_test_misclassified.iloc[i] == 1 else 'Fake'}")
    print(f"Predicted: {'Real' if y_pred_misclassified[i] == 1 else 'Fake'}")
    print(f"Text preview: {X_test_misclassified.iloc[i][:200]}...")

# %% [markdown]
# ## Model Explainability with LIME
#
# Let's use LIME (Local Interpretable Model-agnostic Explanations) to understand what words are driving our predictions.

# %%
# Import LIME
from lime.lime_text import LimeTextExplainer

# Define prediction function for LIME (same preprocessing as training)
def predict_proba_lime(texts):
    # texts is a list of raw text strings
    cleaned = [preprocess_pipeline(text) for text in texts]
    vectorized = tfidf.transform(cleaned)
    return lr_model.predict_proba(vectorized)

# Create explainer
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

# Explain a few predictions
print("Explaining predictions for a few test samples...")
for i in range(min(5, len(X_test))):
    exp = explainer.explain_instance(
        X_test.iloc[i],
        predict_proba_lime,
        num_features=6
    )
    print(f"\nSample {i+1}:")
    print(f"True label: {'Real' if y_test.iloc[i] == 1 else 'Fake'}")
    print(f"Predicted: {'Real' if lr_model.predict(X_test_tfidf[i:i+1])[0] == 1 else 'Fake'}")
    print("Top words contributing to prediction:")
    for word, weight in exp.as_list():
        print(f"  {word}: {weight:.4f}")

# %% [markdown]
# # Saving the Model and Vectorizer
#
# For deployment, we'll save our trained model and TF-IDF vectorizer.

# %%
# Import joblib for saving
import joblib

# Save the model and vectorizer
joblib.dump(lr_model, 'src/fake_news_model.pkl')
joblib.dump(tfidf, 'src/tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully!")
