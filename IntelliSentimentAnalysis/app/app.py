import streamlit as st
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_text

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("📌 About")

st.sidebar.success("Model Accuracy\n\n89%")

st.sidebar.info("""
This project classifies IMDb movie reviews into:

✅ Positive

❌ Negative

### Machine Learning Model
- Logistic Regression

### Feature Extraction
- TF-IDF Vectorizer

### Dataset
IMDb Movie Reviews (50,000 Reviews)

### Developed Using
- Python
- Scikit-Learn
- Streamlit
- Joblib
""")

from pathlib import Path

# -------------------------------------------------
# Load Saved Model (Cached)
# -------------------------------------------------

@st.cache_resource
def load_models():

    importance_path = project_root / "models" / "word_importance.pkl"
    importance_df = joblib.load(importance_path)

    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "sentiment_model.pkl"
    tfidf_path = project_root / "models" / "tfidf_vectorizer.pkl"

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    return model, tfidf, importance_df

model, tfidf, importance_df = load_models()

# -------------------------------------------------
# Main Page
# -------------------------------------------------

st.title("🎬 Movie Review Sentiment Analyzer")

st.write("""
Analyze whether a movie review expresses a **Positive** or **Negative**
sentiment using **Natural Language Processing (NLP)** and
**Machine Learning**.
""")

st.divider()

# -------------------------------------------------
# User Input
# -------------------------------------------------

review = st.text_area(
    "Enter your movie review",
    height=180,
    placeholder="""Example:

I absolutely loved this movie.
The acting was fantastic and the ending was amazing!
"""
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔍 Predict Sentiment"):

    with st.spinner("Analyzing Review..."):

        if len(review.strip()) < 10:

            st.warning("⚠️ Please enter a longer review (minimum 10 characters).")

        else:

            # -----------------------------
            # Preprocess Review
            # -----------------------------

            clean_review = preprocess_text(review)

            with st.expander("🧹 See Preprocessed Review"):
                st.write(clean_review)

            # -----------------------------
            # Convert to TF-IDF
            # -----------------------------

            review_vector = tfidf.transform([clean_review])

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction = model.predict(review_vector)[0]
            tokens = clean_review.split()
            matched = importance_df[
                 importance_df["word"].isin(tokens)
            ].copy()
            matched = matched.sort_values(
                by="weight",
                ascending=False
                )
            probabilities = model.predict_proba(review_vector)[0]

            negative_prob = probabilities[0] * 100
            positive_prob = probabilities[1] * 100

            confidence = max(negative_prob, positive_prob)

            # -----------------------------
            # Prediction Result
            # -----------------------------

            st.subheader("Prediction")

            if prediction == "positive":
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

            st.write(f"### Confidence: {confidence:.2f}%")

            st.divider()
            st.subheader("🧠 Why was this predicted?")
            positive_words = matched[matched["weight"] > 0].head(5)
            negative_words = matched[matched["weight"] < 0].tail(5)
            col1, col2 = st.columns(2)
            with col1:
                st.success("Positive Words")
                if len(positive_words):
                    for word in positive_words["word"]:
                        st.write(f"✅ {word}")
                else:
                    st.write("None")
            with col2:
                st.error("Negative Words")
                if len(negative_words):
                    for word in negative_words["word"]:
                        st.write(f"❌ {word}")
                else:
                    st.write("None")

            # -----------------------------
            # Prediction Probabilities
            # -----------------------------

            st.subheader("Prediction Probabilities")

            st.write(f"🟢 Positive: {positive_prob:.2f}%")
            st.progress(positive_prob / 100)

            st.write(f"🔴 Negative: {negative_prob:.2f}%")
            st.progress(negative_prob / 100)

            st.divider()

            # -----------------------------
            # Review Statistics
            # -----------------------------

            st.subheader("📊 Review Statistics")

            words = len(review.split())
            characters = len(review)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("📝 Words", words)

            with col2:
                st.metric("🔤 Characters", characters)

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Developed by Meghana | Sentiment Analysis using NLP & Machine Learning"
)