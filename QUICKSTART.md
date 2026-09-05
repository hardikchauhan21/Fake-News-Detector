# Quick Start Guide

## Prerequisites
- Python 3.8+
- Git (optional)
- Kaggle account (Optional) Kaggle account (to download dataset)

## Setup

### 1. Clone or create the project
```bash
# If you haven't already created the project:
mkdir fake-news-detector && cd fake-news-detector
# (Files are already created if following along)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click "Download" (you'll need a Kaggle account)
3. Extract the ZIP file
4. Copy `Fake.csv` and `True.csv` to the `data/` directory

### 4. Train the model and generate explanations
```bash
# Launch Jupyter notebook
jupyter notebook

# Open notebooks/01_eda_and_modeling.ipynb
# Run all cells to:
# - Load and explore data
# - Preprocess text
# - Train Logistic Regression model
# - Save model and vectorizer to src/
# - See LIME explainability section (new!) to understand word-level predictions
```

### 5. Run the Streamlit app with explainability
```bash
streamlit run app.py
```

### 6. Use the app
- Open your browser to http://localhost:8501
- Enter news text in the text area
- Click "Analyze News" to get a prediction
- Scroll down to see "Explanation: Why this prediction?" showing top words influencing the decision

## Project Structure
```
fake-news-detector/
├── data/                 # Dataset (Fake.csv, True.csv)
├── notebooks/            # Jupyter notebooks
│   └── 01_eda_and_modeling.ipynb   # Now includes LIME explainability section
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── fake_news_model.pkl       # Trained model (after training)
│   └── tfidf_vectorizer.pkl      # Fitted vectorizer (after training)
├── app.py                # Streamlit application WITH explainability
├── requirements.txt      # Python dependencies (now includes lime)
├── README.md             # Project overview
├── QUICKSTART.md         # This file
├── PROJECT_SUMMARY.md    # Summary of created files
└── .gitignore            # Git ignore file
```

## Troubleshooting

### "Module not found" errors
Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Model files not found
You need to run the notebook first to train and save the model files.

### Dataset loading issues
Ensure Fake.csv and True.csv are in the data/ directory and have the expected columns:
- title
- text
- subject
- date

### Streamlit port already in use
Try a different port:
```bash
streamlit run app.py --server.port=8502
```

### LIME explanation errors
If you see errors in the explanation section, try:
1. Make sure you ran all notebook cells to save the model
2. Check that lime is installed: `pip list | grep lime`
3. The explanation may fail on very short texts - try longer news snippets

## Next Steps for Improvement
1. Experiment with different ML models (Random Forest, SVM) and compare explanations
2. Try deep learning approaches (LSTM, BERT) with attention-based explainability
3. Add more sophisticated explainability methods (SHAP, Integrated Gradients)
4. Deploy to cloud (Heroku, AWS, or Streamlit Community Cloud)
5. Add URL scraping to analyze news from links
6. Implement batch processing for multiple articles

## Learning Resources
- [LIME Documentation](https://lime-ml.readthedocs.io/en/latest/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---
*Ready to showcase in your AI/ML fresher portfolio with explainability!*