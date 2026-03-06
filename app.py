"""
Review Rating Predictor - Streamlit Web Application

A hybrid sentiment analysis system that combines:
1. LSTM Deep Learning Model: Learns patterns from training data
2. VADER Sentiment Analysis: Rule-based sentiment polarity detection

This hybrid approach corrects for noisy labels in training data and improves
prediction accuracy on reviews with negation and strong sentiment words.

Features:
- Preserves negation words (not, never, no) for correct polarity detection
- Highlights sentiment words for interpretability
- Displays sentiment gauge visualization (Plotly)
- Shows word importance and sentiment analysis breakdown
- Provides detailed prediction confidence scores

Author: AI Assistant
Updated: March 2026
"""

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go

import os
import requests

MODEL_PATH = "model/lstm_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=17ZFfBlksi6Svlfhi8Khh0FBQHivlBTp-"

    os.makedirs("model", exist_ok=True)

    r = requests.get(url)

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# =============================================================================
# INITIALIZATION AND CACHING
# =============================================================================

@st.cache_resource
def download_nltk_resources():
    """Download NLTK stopwords and VADER lexicon if not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon')


download_nltk_resources()


@st.cache_resource
def get_custom_stopwords():
    """
    Create a custom stopword list that preserves critical sentiment words.

    IMPORTANT: We MUST preserve:
    1. Negation words (not, no, never, etc.) - Crucial for reversing sentiment polarity
       Example: "not good" means bad (negative), not positive
    2. Sentiment words (bad, good, terrible, amazing) - Carry the core emotion
       Example: Removing bad from bad product removes the sentiment entirely

    These words would typically be removed by standard NLTK stopwords,
    but keeping them dramatically improves sentiment analysis accuracy.

    Returns:
        set: Custom stopwords that exclude negation and sentiment words
    """
    nltk_stopwords = set(stopwords.words('english'))

    # Words that MUST be preserved for sentiment analysis
    negation_words = {
        'not', 'no', 'never', 'nor', 'dont',
        'didnt', 'wont', 'cant', 'shouldnt',
        'wouldnt', 'couldnt', 'havent', 'hasnt',
        'arent', 'isnt', 'wasnt', 'werent'
    }

    sentiment_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst', 'poor',
        'disappointing', 'useless', 'pathetic', 'excellent', 'amazing',
        'great', 'fantastic', 'wonderful', 'good', 'positive', 'lovely',
        'nice', 'perfect', 'awesome'
    }

    preserve_words = negation_words | sentiment_words
    custom_stopwords = nltk_stopwords - preserve_words

    return custom_stopwords


@st.cache_resource
def load_model_and_artifacts():
    """
    Load the trained LSTM model, tokenizer, and label encoder.
    These are cached to avoid reloading on every page interaction.

    Returns:
        tuple: (model, tokenizer, label_encoder)
    """
    try:
        model = tf.keras.models.load_model('model/lstm_model.h5')

        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.error("Please ensure the following files exist:")
        st.error("  - model/lstm_model.h5")
        st.error("  - model/tokenizer.pkl")
        st.error("  - model/label_encoder.pkl")
        st.stop()


@st.cache_resource
def get_sentiment_analyzer():
    """
    Initialize and return the VADER SentimentIntensityAnalyzer.

    VADER (Valence Aware Dictionary and Sentiment Reasoner) is specifically
    designed for social media and product reviews.

    Returns:
        SentimentIntensityAnalyzer: Initialized VADER analyzer
    """
    return SentimentIntensityAnalyzer()


# =============================================================================
# TEXT PROCESSING AND SENTIMENT ANALYSIS
# =============================================================================

def preprocess_text(text):
    """
    Preprocess review text while preserving critical sentiment markers.

    Steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters
    3. Remove stopwords EXCEPT negation and sentiment words

    Why preserve negation and sentiment words:
    - Negation words reverse sentiment polarity ("not good" becomes negative)
    - Sentiment words carry the core emotion ("bad product" vs "product")

    Args:
        text (str): Raw review text

    Returns:
        str: Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    custom_stopwords = get_custom_stopwords()
    words = text.split()
    text = ' '.join([word for word in words if word not in custom_stopwords])

    return text


def compute_sentiment_score(review_text):
    """
    Compute sentiment polarity score using VADER sentiment analysis.

    VADER returns compound score from -1 (most negative) to +1 (most positive).
    This provides a linguistic signal independent of the LSTM model.

    Args:
        review_text (str): Raw review text

    Returns:
        dict: Dictionary with sentiment scores including 'compound', 'pos', 'neg', 'neu'
    """
    analyzer = get_sentiment_analyzer()
    sentiment_scores = analyzer.polarity_scores(review_text)
    return sentiment_scores


def adjust_rating_with_sentiment(predicted_rating, sentiment_scores):
    """
    Adjust LSTM predicted rating based on VADER sentiment analysis.

    HYBRID APPROACH RATIONALE:
    The LSTM model learns from training data, but this data often contains
    noisy labels. By combining LSTM predictions with rule-based VADER sentiment,
    we gain the benefits of both approaches:
    - LSTM: Learns complex patterns and nuances
    - VADER: Detects obvious negative/positive signals

    Adjustment thresholds:
    - compound < -0.5 (very negative): Reduce by 2 stars
    - -0.5 <= compound < -0.1 (negative): Reduce by 1 star
    - -0.1 <= compound <= 0.1 (neutral): No adjustment
    - 0.1 < compound <= 0.5 (positive): No adjustment
    - compound > 0.5 (very positive): Increase by 1 star (max 5)

    Args:
        predicted_rating (float or int): Original LSTM predicted rating
        sentiment_scores (dict): Dictionary from VADER with 'compound' key

    Returns:
        tuple: (adjusted_rating, adjustment_amount, adjustment_reason)
    """
    compound = sentiment_scores.get('compound', 0)
    adjustment = 0
    reason = "No adjustment"

    if compound < -0.5:
        adjustment = -2
        reason = f"Strong negative sentiment detected (score: {compound:.2f})"
    elif -0.5 <= compound < -0.1:
        adjustment = -1
        reason = f"Moderate negative sentiment detected (score: {compound:.2f})"
    elif -0.1 <= compound <= 0.1:
        adjustment = 0
        reason = f"Neutral sentiment (score: {compound:.2f})"
    elif compound > 0.5:
        adjustment = 1
        reason = f"Strong positive sentiment detected (score: {compound:.2f})"

    adjusted_rating = max(1, min(5, predicted_rating + adjustment))

    return adjusted_rating, adjustment, reason


def get_sentiment_category(sentiment_scores):
    """
    Categorize sentiment as Negative, Neutral, or Positive for UI display.

    Args:
        sentiment_scores (dict): Dictionary from VADER with 'compound' key

    Returns:
        str: One of "Negative", "Neutral", or "Positive"
    """
    compound = sentiment_scores.get('compound', 0)
    if compound < -0.1:
        return "Negative"
    elif compound > 0.1:
        return "Positive"
    else:
        return "Neutral"


def highlight_negation_words(text):
    """
    Highlight negation words in preprocessed text for visualization.

    Helps users understand which words are used for sentiment reversal.

    Args:
        text (str): Preprocessed text

    Returns:
        str: Text with negation words highlighted in markdown
    """
    negation_words = {
        'not', 'no', 'never', 'nor', 'dont',
        'didnt', 'wont', 'cant', 'shouldnt',
        'wouldnt', 'couldnt', 'havent', 'hasnt',
        'arent', 'isnt', 'wasnt', 'werent'
    }

    words = text.split()
    highlighted = [
        f"**_{word}_**" if word in negation_words else word
        for word in words
    ]

    return ' '.join(highlighted)


def create_sentiment_gauge(compound_score):
    """
    Create a Plotly gauge chart to visualize VADER sentiment compound score.

    VISUALIZATION PURPOSE:
    Provides an intuitive visual representation of sentiment polarity:
    - Red (left, -1.0): Strongly negative
    - Yellow (middle, 0.0): Neutral
    - Green (right, +1.0): Strongly positive

    This helps users understand the linguistic sentiment independent of
    the LSTM model's learned associations.

    Args:
        compound_score (float): VADER compound sentiment score (-1 to +1)

    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    if compound_score < -0.3:
        color = "red"
    elif compound_score > 0.3:
        color = "green"
    else:
        color = "gold"

    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number",
        value=compound_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Intensity"},
        gauge={
            'axis': {
                'range': [-1, 1],
                'tickvals': [-1, -0.5, 0, 0.5, 1],
                'ticktext': ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            },
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.3], 'color': "rgba(255, 0, 0, 0.1)"},
                {'range': [-0.3, 0.3], 'color': "rgba(255, 255, 0, 0.1)"},
                {'range': [0.3, 1], 'color': "rgba(0, 128, 0, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "gray", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        },
    )])

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=70, b=20),
        font=dict(size=12)
    )

    return fig


def highlight_sentiment_words(review_text):
    """
    Highlight sentiment words in the review text for interpretability.

    INTERPRETABILITY PURPOSE:
    Shows which specific words in the review triggered sentiment detection.
    By highlighting emotional language, we make the model decision process
    more transparent.

    Args:
        review_text (str): Raw review text

    Returns:
        str: HTML string with highlighted sentiment words
    """
    negative_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst',
        'disappointing', 'useless', 'pathetic', 'poor', 'shame',
        'worse', 'badly', 'disgusting', 'nasty', 'waste',
        'garbage', 'junk', 'annoying', 'frustrating'
    }

    positive_words = {
        'good', 'excellent', 'amazing', 'great', 'fantastic',
        'wonderful', 'awesome', 'perfect', 'lovely', 'nice',
        'beautiful', 'brilliant', 'outstanding', 'impressive',
        'superb', 'terrific', 'exceptional', 'marvelous', 'delightful'
    }

    words = review_text.split()
    highlighted_words = []

    for word in words:
        word_clean = re.sub(r'[^\w]', '', word.lower())

        if word_clean in negative_words:
            highlighted_words.append(f"<span style='color:red;font-weight:bold'>{word}</span>")
        elif word_clean in positive_words:
            highlighted_words.append(f"<span style='color:green;font-weight:bold'>{word}</span>")
        else:
            highlighted_words.append(word)

    return ' '.join(highlighted_words)


def get_detected_sentiment_words(review_text):
    """
    Extract and categorize sentiment words found in the review.

    Helps users see exactly which sentiment words influenced the analysis.

    Args:
        review_text (str): Raw review text

    Returns:
        tuple: (negative_words_found, positive_words_found)
    """
    negative_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst',
        'disappointing', 'useless', 'pathetic', 'poor',
        'shame', 'worse', 'badly', 'disgusting', 'nasty',
        'waste', 'garbage', 'junk', 'annoying', 'frustrating'
    }

    positive_words = {
        'good', 'excellent', 'amazing', 'great', 'fantastic',
        'wonderful', 'awesome', 'perfect', 'lovely', 'nice',
        'beautiful', 'brilliant', 'outstanding', 'impressive',
        'superb', 'terrific', 'exceptional', 'marvelous', 'delightful'
    }

    words = review_text.lower().split()
    detected_negative = []
    detected_positive = []

    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in negative_words and word_clean not in detected_negative:
            detected_negative.append(word_clean)
        elif word_clean in positive_words and word_clean not in detected_positive:
            detected_positive.append(word_clean)

    return detected_negative, detected_positive


# =============================================================================
# MODEL PREDICTION
# =============================================================================

def predict_rating(review_text, model, tokenizer, label_encoder):
    """
    Predict review rating using hybrid LSTM + VADER sentiment analysis.

    This is a HYBRID APPROACH combining two complementary methods:
    1. LSTM deep learning: Learns complex patterns from training data
    2. VADER sentiment: Rule-based polarity detection
    3. Adjustment: VADER adjusts LSTM predictions if needed

    Why hybrid? Training data often contains noisy labels. Combining both
    approaches gives us the benefits of learned patterns + linguistic rules.

    Args:
        review_text (str): The product review text
        model: Trained LSTM model
        tokenizer: Keras tokenizer fitted on training data
        label_encoder: Scikit-learn label encoder for rating classes

    Returns:
        dict: Contains lstm_rating, confidence, sentiment_scores, final_rating, etc.
    """
    # Get LSTM prediction
    processed_text = preprocess_text(review_text)
    sequence = tokenizer.texts_to_sequences([processed_text])

    max_length = 250
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction = model.predict(padded_sequence, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    lstm_rating = label_encoder.inverse_transform([predicted_class])[0]

    # Get probability distribution
    all_classes = label_encoder.classes_
    probabilities = {
        str(rating): float(prob) * 100
        for rating, prob in zip(all_classes, prediction[0])
    }

    # Get VADER sentiment scores
    sentiment_scores = compute_sentiment_score(review_text)

    # Adjust rating based on sentiment
    adjusted_rating, adjustment, reason = adjust_rating_with_sentiment(
        lstm_rating, sentiment_scores
    )

    sentiment_category = get_sentiment_category(sentiment_scores)

    return {
        'lstm_rating': lstm_rating,
        'lstm_confidence': confidence,
        'sentiment_scores': sentiment_scores,
        'sentiment_category': sentiment_category,
        'final_rating': adjusted_rating,
        'adjustment': adjustment,
        'adjustment_reason': reason,
        'probabilities': probabilities,
        'processed_text': processed_text
    }


# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application with hybrid LSTM + VADER sentiment analysis."""

    # Page configuration
    st.set_page_config(
        page_title="Review Rating Predictor",
        page_icon="⭐",
        layout="wide"
    )

    # Sidebar: Model information
    with st.sidebar:
        st.title("📚 About This App")

        st.markdown("""
        ### How It Works

        This app uses a **Hybrid Sentiment Analysis** system:

        **LSTM Neural Network**
        - Trained on thousands of product reviews
        - Learns complex patterns and sentiment relationships
        - Generates initial rating prediction

        **VADER Sentiment Analysis**
        - Rule-based approach optimized for reviews
        - Detects strong negative/positive signals
        - Adjusts LSTM predictions for correction

        ### Key Features

        ✨ **Smart Preprocessing**
        - Preserves negation words (not, never, no)
        - Preserves sentiment words (bad, good, amazing)
        - Removes only generic stopwords

        📊 **Advanced Visualizations**
        - Sentiment intensity gauge (Plotly)
        - Word importance highlighting
        - Probability distribution charts

        🔍 **Full Transparency**
        - Shows LSTM prediction + confidence
        - Shows VADER sentiment analysis
        - Explains all rating adjustments

        ### Model Details

        - **Architecture**: Bidirectional LSTM
        - **Vocab Size**: 20,000 words
        - **Sequence Length**: 250 tokens
        - **Training**: Balanced dataset, 12 epochs
        - **Metrics**: Accuracy, Precision, Recall

        ### Preserved Words

        **Negation**: not, no, never, nor, dont, wont, cant
        **Sentiment**: bad, awful, terrible, excellent, amazing, great
        """)

        st.divider()
        st.markdown("**Built with Streamlit, TensorFlow & NLTK**")

    # Main content
    st.title("⭐ Review Rating Predictor")

    st.markdown("""
    Enter a product review below, and our hybrid LSTM + VADER system will predict
    the rating (1-5 stars). The system combines deep learning with rule-based
    sentiment analysis for improved accuracy.
    """)

    # Load model and artifacts
    model, tokenizer, label_encoder = load_model_and_artifacts()

    # Example reviews section
    st.markdown("### 💡 Try These Examples")
    col1, col2, col3 = st.columns(3)

    example_reviews = {
        "Negative": "not good, never buying again",
        "Positive": "amazing product, excellent quality",
        "Mixed": "good quality but bad customer service"
    }

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    with col1:
        if st.button("📉 Negative Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Negative"]
            st.rerun()

    with col2:
        if st.button("📈 Positive Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Positive"]
            st.rerun()

    with col3:
        if st.button("↔️ Mixed Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Mixed"]
            st.rerun()

    st.divider()

    # User input
    review_input = st.text_area(
        label="Enter your product review:",
        placeholder="Example: This product is amazing! Works perfectly.",
        height=150,
        value=st.session_state.review_text
    )

    # Prediction button
    if st.button("🔮 Predict Rating", use_container_width=True):
        if not review_input.strip():
            st.warning("⚠️ Please enter a review text before predicting.")
        else:
            st.session_state.review_text = review_input

            with st.spinner("Analyzing review..."):
                try:
                    result = predict_rating(
                        review_input,
                        model,
                        tokenizer,
                        label_encoder
                    )

                    # Extract results
                    lstm_rating = result['lstm_rating']
                    lstm_confidence = result['lstm_confidence']
                    sentiment_scores = result['sentiment_scores']
                    sentiment_category = result['sentiment_category']
                    final_rating = result['final_rating']
                    adjustment = result['adjustment']
                    adjustment_reason = result['adjustment_reason']
                    probabilities = result['probabilities']
                    processed_text = result['processed_text']

                    st.success("✅ Prediction Complete!")

                    # Main prediction metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="🤖 LSTM Prediction",
                            value=f"{lstm_rating} ⭐",
                            delta=f"{lstm_confidence:.1f}% confidence"
                        )

                    with col2:
                        emoji = "😞" if sentiment_category == "Negative" else (
                            "😊" if sentiment_category == "Positive" else "😐"
                        )
                        st.metric(
                            label="💭 Sentiment",
                            value=f"{emoji} {sentiment_category}",
                            delta=f"Score: {sentiment_scores['compound']:.2f}"
                        )

                    with col3:
                        st.metric(
                            label="✨ Final Rating",
                            value=f"{final_rating} ⭐"
                        )

                    st.divider()

                    # Sentiment adjustment details
                    st.subheader("🔧 Sentiment-Based Adjustment")

                    if adjustment != 0:
                        adjustment_text = f"{adjustment:+d}"
                        st.info(
                            f"**Adjustment:** {adjustment_text} star(s)\n\n"
                            f"{adjustment_reason}"
                        )
                    else:
                        st.info(
                            f"**No adjustment needed**\n\n"
                            f"{adjustment_reason}"
                        )

                    st.divider()

                    # Sentiment analysis breakdown
                    st.subheader("📊 Sentiment Analysis Breakdown")

                    scol1, scol2 = st.columns(2)

                    with scol1:
                        st.metric(
                            "Negative Sentiment",
                            f"{sentiment_scores['neg']*100:.1f}%"
                        )
                        st.metric(
                            "Positive Sentiment",
                            f"{sentiment_scores['pos']*100:.1f}%"
                        )

                    with scol2:
                        st.metric(
                            "Neutral Sentiment",
                            f"{sentiment_scores['neu']*100:.1f}%"
                        )
                        st.metric(
                            "Compound Score",
                            f"{sentiment_scores['compound']:.3f}",
                            delta="(-1 to +1)"
                        )

                    st.divider()

                    # Sentiment gauge visualization
                    st.subheader("🎯 Sentiment Intensity Gauge")
                    gauge_fig = create_sentiment_gauge(sentiment_scores['compound'])
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    st.divider()

                    # Word importance visualization
                    st.subheader("🔑 Key Sentiment Words Detected")

                    detected_neg, detected_pos = get_detected_sentiment_words(review_input)

                    wcol1, wcol2 = st.columns(2)

                    with wcol1:
                        if detected_neg:
                            st.error(f"**Negative Words:** {', '.join(detected_neg)}")
                        else:
                            st.success("**No negative words detected**")

                    with wcol2:
                        if detected_pos:
                            st.success(f"**Positive Words:** {', '.join(detected_pos)}")
                        else:
                            st.error("**No positive words detected**")

                    st.markdown("**Review with Highlighted Sentiment Words:**")
                    highlighted_html = highlight_sentiment_words(review_input)
                    st.markdown(highlighted_html, unsafe_allow_html=True)

                    st.divider()

                    # Probability distribution
                    st.subheader("📈 Prediction Probability Distribution")

                    prob_data = pd.DataFrame({
                        'Rating': list(probabilities.keys()),
                        'Probability (%)': list(probabilities.values())
                    })

                    prob_data['Rating'] = pd.Categorical(
                        prob_data['Rating'],
                        categories=sorted(probabilities.keys()),
                        ordered=True
                    )
                    prob_data = prob_data.sort_values('Rating')

                    st.bar_chart(
                        data=prob_data.set_index('Rating'),
                        height=300,
                        use_container_width=True
                    )

                    st.markdown("### Rating Probabilities")
                    prob_cols = st.columns(len(probabilities))
                    for idx, (rating, prob) in enumerate(sorted(probabilities.items())):
                        with prob_cols[idx]:
                            st.metric(f"{rating} ⭐", f"{prob:.2f}%")

                    st.divider()

                    # Text processing details
                    st.subheader("📋 Text Processing Details")

                    highlighted = highlight_negation_words(processed_text)

                    tcol1, tcol2 = st.columns(2)

                    with tcol1:
                        st.markdown("**Original Text:**")
                        st.text(review_input)

                    with tcol2:
                        st.markdown("**Cleaned Text:**")
                        st.markdown(highlighted)

                    with st.expander("🔍 Stopword Removal Details"):
                        original_words = set(review_input.lower().split())
                        cleaned_words = set(processed_text.split())
                        removed_words = original_words - cleaned_words

                        if removed_words:
                            st.markdown(f"**Removed:** {', '.join(sorted(removed_words))}")
                        else:
                            st.markdown("**No stopwords removed (good!)**")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.error("Please check that all model files are present.")

    # Footer with explanation
    st.divider()
    st.markdown("""
    ---
    ### 🎯 How the Hybrid Approach Works

    **Problem with traditional approaches:**
    - Pure LSTM models can learn biases from noisy training data
    - Pure rule-based systems miss nuanced sentiment
    - Either approach alone is suboptimal

    **Our solution - Hybrid System:**
    1. **LSTM Stage**: Generates initial rating prediction
    2. **VADER Stage**: Analyzes linguistic sentiment independently
    3. **Adjustment Stage**: Combines both signals for final rating

    **Benefits:**
    ✅ Corrects obvious noisy labels in training data
    ✅ Handles negation correctly ("not good" → negative)
    ✅ Transparent and explainable predictions
    ✅ Combines learned patterns + linguistic rules

    **Example:**
    - Review: "not good quality"
    - LSTM alone: 4 stars (lacks negation context)
    - VADER sentiment: -0.45 (detects negative)
    - Final rating: 3 stars ✅

    **Learn more about sentiment analysis:**
    - VADER: https://github.com/cjhutto/vaderSentiment
    - LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """)


if __name__ == "__main__":
    main()
