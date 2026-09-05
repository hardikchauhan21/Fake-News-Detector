"""
Streamlit app for Fake News Detector with LIME explainability
"""
import streamlit as st
import joblib
import numpy as np
import os
from src.data_loader import preprocess_pipeline

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
This app uses machine learning to detect whether a news article is likely real or fake.
Enter a news headline or article text below to get a prediction with explanation.
""")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model_path = "src/fake_news_model.pkl"
    vectorizer_path = "src/tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        return None, None

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.warning("""
    ⚠️ Model files not found!

    To use this app, you need to:
    1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    2. Place Fake.csv and True.csv in the data/ directory
    3. Run the Jupyter notebook to train and save the model
    4. Then restart this app

    For now, you can explore the notebook in the notebooks/ directory.
    """)

    # Show sample prediction interface anyway
    st.info("Below is a demo of how the app will work once the model is trained.")

    # Demo mode - shows what the interface will look like
    user_input = st.text_area(
        "Enter news text here (demo mode):",
        height=150,
        placeholder="Paste a news headline or article text..."
    )

    if st.button("Detect Fake News (Demo)"):
        if user_input.strip():
            st.info("🔍 In demo mode, this would show a prediction with explanation...")
            st.write("**Demo Result:** This is where the prediction would appear")
            st.write("- **Prediction:** Real/Fake")
            st.write("- **Confidence:** XX%")
            st.write("- **Explanation:** Top words contributing to the prediction")
        else:
            st.warning("Please enter some text to analyze.")
else:
    # Main app with actual model
    st.success("✅ Model loaded successfully!")

    # Text input
    user_input = st.text_area(
        "Enter news text here:",
        height=200,
        placeholder="Paste a news headline or article text to analyze..."
    )

    if st.button("Analyze News", type="primary"):
        if user_input.strip():
            # Preprocess the input
            cleaned_text = preprocess_pipeline(user_input)

            # Vectorize
            text_vectorized = vectorizer.transform([cleaned_text])

            # Predict
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = model.predict_proba(text_vectorized)[0]

            # Get confidence
            confidence = max(prediction_proba) * 100

            # Display results
            st.subheader("Analysis Results")

            if prediction == 1:  # Real
                st.success(f"✅ This appears to be **REAL NEWS**")
            else:  # Fake
                st.error(f"❌ This appears to be **FAKE NEWS**")

            st.info(f"🎯 Confidence: {confidence:.1f}%")

            # Show probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake News Probability", f"{prediction_proba[0]*100:.1f}%")
            with col2:
                st.metric("Real News Probability", f"{prediction_proba[1]*100:.1f}%")

            # LIME Explanation
            st.subheader("Explanation: Why this prediction?")
            try:
                from lime.lime_text import LimeTextExplainer

                # Define prediction function for LIME
                def predict_proba_lime(texts):
                    # texts is a list of raw text strings
                    cleaned = [preprocess_pipeline(text) for text in texts]
                    vectorized = vectorizer.transform(cleaned)
                    return model.predict_proba(vectorized)

                # Create explainer
                explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

                # Generate explanation
                exp = explainer.explain_instance(
                    user_input,
                    predict_proba_lime,
                    num_features=10,  # show top 10 words
                    labels=[1] if prediction == 1 else [0]  # explain the predicted class
                )

                # Display explanation as a list
                st.write(f"Top words influencing the prediction for **{'Real' if prediction == 1 else 'Fake'}** news:")
                for word, weight in exp.as_list(label=[1] if prediction == 1 else [0][0]):
                    color = "green" if weight > 0 else "red"
                    st.markdown(f"- <span style='color:{color}'>{word}</span>: {weight:.4f}", unsafe_allow_html=True)

                # Also show as a bar chart (optional)
                # st.pyplot(exp.as_pyplot_figure())

            except Exception as e:
                st.warning(f"Could not generate explanation: {e}")

            # Show processed text (expandable)
            with st.expander("See processed text"):
                st.write("**Original text:**")
                st.write(user_input[:500] + "..." if len(user_input) > 500 else user_input)
                st.write("**Cleaned text:**")
                st.write(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
        else:
            st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
### How it works
1. Text is cleaned (lowercase, punctuation removed, stopwords removed)
2. Converted to TF-IDF features
3. Logistic Regression model predicts real vs fake
4. Confidence score shows model certainty
5. LIME explainability shows which words most influenced the prediction

### Limitations
- Trained on a specific dataset - may not generalize to all news types
- Works best with English news articles
- Should be used as a tool, not definitive truth
- Explanations are approximate and may not capture complex interactions

---
*Built as a portfolio project for AI/ML fresher placements with model explainability*
""")