# Fake News Detector

## Project Overview
An end-to-end NLP project to detect fake news articles using machine learning with model explainability. This project demonstrates the complete ML pipeline: data collection, preprocessing, model training, evaluation, deployment, and explainability.

## Problem Statement
With the rise of misinformation online, there's a growing need for automated tools to help identify potentially fake or misleading news articles. This project builds a classifier that can distinguish between real and fake news based on article text, and provides explanations for its predictions.

## Dataset
- Source: Kaggle "Fake and real news dataset"
- Contains: Real news from Reuters and fake news from various sources
- Features: Article text, title, subject, date
- Target: Label (REAL/FAKE)

## Approach
1. **Data Preprocessing**: Text cleaning (lowercase, punctuation removal, stopwords)
2. **Feature Engineering**: TF-IDF vectorization
3. **Model Training**: Logistic Regression baseline
4. **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
5. **Explainability**: LIME (Local Interpretable Model-agnostic Explanations) to understand word-level contributions
6. **Deployment**: Simple Streamlit web interface for real-time prediction with explanations

## Files
- `data/` - Contains dataset (fake.csv, true.csv)
- `notebooks/` - Jupyter notebooks for EDA and modeling (now includes LIME explainability)
- `src/` - Source code for preprocessing and prediction
- `app.py` - Streamlit application with explainability
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `QUICKSTART.md` - Setup guide
- `PROJECT_SUMMARY.md` - Summary of what was created

## Results
- Baseline Logistic Regression accuracy: ~95%
- Model generalizes well to unseen data
- Streamlit app provides real-time predictions with confidence scores and word-level explanations

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset from Kaggle and place in data/
3. Train model: Run `jupyter notebook` and execute all cells in `notebooks/01_eda_and_modeling.ipynb`
4. Run Streamlit app: `streamlit run app.py`
5. Enter news text to get prediction with explanation

## Learning Outcomes
- Text preprocessing techniques (NLTK/spaCy)
- Feature engineering with TF-IDF
- Model evaluation beyond accuracy
- End-to-end ML pipeline implementation
- Model explainability with LIME
- Basic web deployment with Streamlit

## Future Improvements
- Add deep learning models (LSTM, BERT)
- Compare multiple algorithms with explainability
- Add URL scraping for real-time news checking
- Deploy to cloud (Heroku/AWS)

---
*Built as a portfolio project for AI/ML fresher placements*