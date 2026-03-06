"""
Review Rating Predictor - Streamlit Web Application
This app loads a trained LSTM model to predict product review ratings.

IMPROVEMENTS:
- Enhanced preprocessing that preserves negation words and sentiment words
- Custom stopword list that excludes critical sentiment-related terms
- Better handling of sentiment analysis through selective stopword removal
- HYBRID APPROACH: Combines LSTM predictions with VADER sentiment analysis
  * LSTM captures learned rating patterns from training data
  * VADER provides rule-based sentiment polarity as a correction signal
  * This approach corrects for noisy labels in training data
- ADVANCED VISUALIZATIONS for interpretability and transparency:
  * Sentiment Gauge: Plotly gauge chart showing compound sentiment score (-1 to +1)
  * Word Importance: Highlights detected sentiment words in red (negative) and green (positive)
  * These features help users understand why the model made specific predictions
"""

# Import required libraries
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

# Download NLTK resources (only once)
@st.cache_resource
def download_nltk_resources():
    """Download NLTK stopwords and VADER lexicon if not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        # Test VADER availability
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_resources()

# Create custom stopword list
@st.cache_resource
def get_custom_stopwords():
    """
    Create a custom stopword list that preserves critical words for sentiment analysis.
    
    IMPORTANT: We MUST preserve:
    1. Negation words (not, no, never, etc.) - These are crucial for reversing sentiment
       Example: "not good" means bad, "no problem" means positive
    2. Sentiment words (bad, good, terrible, etc.) - These carry the core sentiment
       Example: Removing "bad" from "bad product" removes the sentiment entirely
    
    These words would be removed by standard NLTK stopwords, but keeping them
    dramatically improves sentiment analysis accuracy.
    
    Returns:
        set: Custom stopwords that exclude negation and sentiment words
    """
    # Start with NLTK stopwords
    nltk_stopwords = set(stopwords.words('english'))
    
    # Words that MUST be preserved for sentiment analysis
    negation_words = {
        'not', 'no', 'never', 'nor', "don't", "didn't", 
        "won't", "can't", "shouldn't", "wouldn't", "couldn't", 
        "haven't", "hasn't", "aren't", "aren", "wasn", "weren",
        "isn", "ain", "doesn"
    }
    
    sentiment_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst', 'poor', 
        'disappointing', 'disappointing', 'useless', 'pathetic',
        'excellent', 'amazing', 'great', 'fantastic', 'wonderful',
        'good', 'positive', 'lovely', 'nice', 'perfect', 'awesome'
    }
    
    # Combine all words to preserve
    preserve_words = negation_words | sentiment_words
    
    # Remove preserved words from stopwords
    custom_stopwords = nltk_stopwords - preserve_words
    
    return custom_stopwords

# Load model, tokenizer, and label encoder (cached for performance)
@st.cache_resource
def load_model_and_artifacts():
    """
    Load the trained LSTM model, tokenizer, and label encoder.
    These are cached to avoid reloading on every page interaction.
    """
    try:
        # Load the trained LSTM model
        model = tf.keras.models.load_model('model/lstm_model.h5')
        
        # Load the tokenizer
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load the label encoder
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, tokenizer, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

# Initialize VADER sentiment analyzer (cached for performance)
@st.cache_resource
def get_sentiment_analyzer():
    """
    Initialize and return the VADER SentimentIntensityAnalyzer.
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically 
    designed for social media and product reviews, making it perfect for this task.
    
    Returns:
        SentimentIntensityAnalyzer: Initialized VADER analyzer
    """
    return SentimentIntensityAnalyzer()

def compute_sentiment_score(review_text):
    """
    Compute sentiment polarity score using VADER sentiment analysis.
    
    VADER returns a compound score from -1 (most negative) to +1 (most positive).
    This provides a linguistic signal independent of the LSTM model.
    
    Why VADER?
    - Rule-based approach specifically tuned for social media and reviews
    - Fast and doesn't require training data
    - Excellent at detecting sentiment intensity (exclamation marks, caps, etc.)
    - Complements deep learning models well
    
    Args:
        review_text (str): Raw review text
    
    Returns:
        dict: Dictionary with sentiment scores ('compound', 'positive', 'neutral', 'negative')
    """
    analyzer = get_sentiment_analyzer()
    sentiment_scores = analyzer.polarity_scores(review_text)
    return sentiment_scores

def adjust_rating_with_sentiment(predicted_rating, sentiment_scores):
    """
    Adjust the LSTM predicted rating based on VADER sentiment analysis.
    
    HYBRID APPROACH RATIONALE:
    ===========================
    The LSTM model learns from training data, but training data often contains noisy labels.
    For example, a review might have a 5-star rating but contain very negative language.
    
    By combining LSTM predictions with rule-based VADER sentiment:
    - We get the learned patterns from LSTM (good for nuanced cases)
    - We get the linguistic rules from VADER (good for obvious negative/positive cases)
    - Together, they're more robust than either alone
    
    Adjustment thresholds:
    - compound < -0.5 (very negative): Reduce by 2 stars
    - -0.5 <= compound < -0.1 (somewhat negative): Reduce by 1 star
    - -0.1 <= compound <= 0.1 (neutral): No change
    - 0.1 < compound <= 0.5 (somewhat positive): No change (LSTM already captures this)
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
    
    # Strong negative sentiment - reduce rating significantly
    if compound < -0.5:
        adjustment = -2
        reason = f"Strong negative sentiment detected (score: {compound:.2f})"
    
    # Moderate negative sentiment - reduce rating slightly
    elif -0.5 <= compound < -0.1:
        adjustment = -1
        reason = f"Moderate negative sentiment detected (score: {compound:.2f})"
    
    # Neutral sentiment - no adjustment
    elif -0.1 <= compound <= 0.1:
        adjustment = 0
        reason = f"Neutral sentiment (score: {compound:.2f})"
    
    # Strong positive sentiment - increase rating
    elif compound > 0.5:
        adjustment = 1
        reason = f"Strong positive sentiment detected (score: {compound:.2f})"
    
    # Apply adjustment and clamp to valid range [1, 5]
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

# Text preprocessing function with improved stopword handling
def preprocess_text(text):
    """
    Preprocess the input text using improved steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters (keep only letters and spaces)
    3. Remove English stopwords WHILE PRESERVING negation and sentiment words
    
    Why this matters:
    - Negation words (not, never, no) are critical for understanding sentiment polarity
    - Sentiment words (bad, good, awful, amazing) carry the core meaning
    - Standard preprocessing removes these, losing important information
    
    Args:
        text (str): Raw input text to preprocess
    
    Returns:
        str: Preprocessed text with sentiment-critical words preserved
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords EXCEPT negation and sentiment words
    custom_stopwords = get_custom_stopwords()
    words = text.split()
    text = ' '.join([word for word in words if word not in custom_stopwords])
    
    return text

def highlight_negation_words(text):
    """
    Highlight negation words in the preprocessed text for visualization.
    This helps users understand which words are being used for sentiment reversal.
    
    Args:
        text (str): Preprocessed text
    
    Returns:
        str: Text with negation words highlighted in markdown
    """
    negation_words = {'not', 'no', 'never', 'nor', "don't", "didn't", 
                     "won't", "can't", "shouldn't", "wouldn't", "couldn't", 
                     "haven't", "hasn't", "aren't", "isn't", "wasn't", "weren't"}
    
    words = text.split()
    highlighted = []
    for word in words:
        if word in negation_words:
            highlighted.append(f"**_{word}_**")  # Bold and italic for negation words
        else:
            highlighted.append(word)
    
    return ' '.join(highlighted)

def create_sentiment_gauge(compound_score):
    """
    Create a Plotly gauge chart to visualize VADER sentiment compound score.
    
    VISUALIZATION PURPOSE:
    This gauge provides an intuitive visual representation of sentiment polarity:
    - Red (left, -1.0): Strongly negative sentiment
    - Yellow (middle, 0.0): Neutral sentiment
    - Green (right, +1.0): Strongly positive sentiment
    
    The gauge helps users quickly understand the linguistic sentiment of a review,
    which is independent of the LSTM model's learned associations. This allows users
    to understand why the hybrid approach makes certain adjustments.
    
    Args:
        compound_score (float): VADER compound sentiment score (-1 to +1)
    
    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    # Determine color based on sentiment range
    if compound_score < -0.3:
        color = "red"
    elif compound_score > 0.3:
        color = "green"
    else:
        color = "gold"  # Yellow for neutral
    
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
                {'range': [-1, -0.3], 'color': "rgba(255, 0, 0, 0.1)"},      # Red zone (negative)
                {'range': [-0.3, 0.3], 'color': "rgba(255, 255, 0, 0.1)"},   # Yellow zone (neutral)
                {'range': [0.3, 1], 'color': "rgba(0, 128, 0, 0.1)"}         # Green zone (positive)
            ],
            'threshold': {
                'line': {'color': "gray", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        },
        number={'suffix': "", 'decimals': 2}
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
    This visualization helps users understand which specific words in their review
    triggered sentiment detection. By highlighting emotional language, we make the
    model's decision process more transparent and understandable.
    
    The VADER sentiment analyzer uses these words in its compound score calculation,
    so highlighting them explains why the review got a certain sentiment score.
    
    Args:
        review_text (str): Raw review text
    
    Returns:
        str: HTML string with highlighted sentiment words
    """
    # Define sentiment word sets
    negative_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst', 'disappointing',
        'useless', 'pathetic', 'poor', 'shame', 'worse', 'badly', 'disgusting',
        'nasty', 'waste', 'garbage', 'junk', 'annoying', 'frustrating'
    }
    
    positive_words = {
        'good', 'excellent', 'amazing', 'great', 'fantastic', 'wonderful',
        'awesome', 'perfect', 'lovely', 'nice', 'beautiful', 'brilliant',
        'outstanding', 'impressive', 'superb', 'terrific', 'fantastic',
        'exceptional', 'marvelous', 'delightful'
    }
    
    # Convert text to lowercase for matching but preserve original for display
    text_lower = review_text.lower()
    words = review_text.split()
    
    highlighted_words = []
    processed_lower = []
    
    for word in words:
        # Remove punctuation for matching but keep it for display
        word_clean = re.sub(r'[^\w]', '', word.lower())
        
        if word_clean in negative_words:
            # Highlight in red with bold styling
            highlighted_words.append(f"<span style='color:red;font-weight:bold'>{word}</span>")
        elif word_clean in positive_words:
            # Highlight in green with bold styling
            highlighted_words.append(f"<span style='color:green;font-weight:bold'>{word}</span>")
        else:
            highlighted_words.append(word)
    
    # Join words back together, preserving spaces
    highlighted_text = ' '.join(highlighted_words)
    
    return highlighted_text

def get_detected_sentiment_words(review_text):
    """
    Extract and categorize sentiment words found in the review.
    
    Returns two lists:
    1. Negative sentiment words detected
    2. Positive sentiment words detected
    
    This helps users see exactly which sentiment words influenced the analysis.
    
    Args:
        review_text (str): Raw review text
    
    Returns:
        tuple: (negative_words_found, positive_words_found)
    """
    negative_words = {
        'bad', 'awful', 'terrible', 'horrible', 'worst', 'disappointing',
        'useless', 'pathetic', 'poor', 'shame', 'worse', 'badly', 'disgusting',
        'nasty', 'waste', 'garbage', 'junk', 'annoying', 'frustrating'
    }
    
    positive_words = {
        'good', 'excellent', 'amazing', 'great', 'fantastic', 'wonderful',
        'awesome', 'perfect', 'lovely', 'nice', 'beautiful', 'brilliant',
        'outstanding', 'impressive', 'superb', 'terrific', 'fantastic',
        'exceptional', 'marvelous', 'delightful'
    }
    
    words = review_text.lower().split()
    
    # Extract sentiment words
    detected_negative = []
    detected_positive = []
    
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in negative_words and word_clean not in detected_negative:
            detected_negative.append(word_clean)
        elif word_clean in positive_words and word_clean not in detected_positive:
            detected_positive.append(word_clean)
    
    return detected_negative, detected_positive


def predict_rating(review_text, model, tokenizer, label_encoder):
    """
    Predict the rating for a given review using LSTM model + VADER sentiment analysis.

    Hybrid approach:
    1. LSTM predicts rating based on learned patterns
    2. VADER analyzes sentiment polarity
    3. Sentiment adjusts the rating if necessary
    """

    # Preprocess text
    processed_text = preprocess_text(review_text)
    sequence = tokenizer.texts_to_sequences([processed_text])

    max_length = 250
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction = model.predict(padded_sequence, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    lstm_rating = label_encoder.inverse_transform([predicted_class])[0]

    # Probability distribution
    all_classes = label_encoder.classes_
    probabilities = {
        str(rating): float(prob) * 100
        for rating, prob in zip(all_classes, prediction[0])
    }

    # Sentiment analysis
    sentiment_scores = compute_sentiment_score(review_text)

    adjusted_rating, adjustment, reason = adjust_rating_with_sentiment(
        lstm_rating,
        sentiment_scores
    )

    sentiment_category = get_sentiment_category(sentiment_scores)

    return {
        "lstm_rating": lstm_rating,
        "lstm_confidence": confidence,
        "sentiment_scores": sentiment_scores,
        "sentiment_category": sentiment_category,
        "final_rating": adjusted_rating,
        "adjustment": adjustment,
        "adjustment_reason": reason,
        "probabilities": probabilities,
        "processed_text": processed_text
    }
    
    This is a HYBRID APPROACH:
    1. Run LSTM deep learning model to get initial prediction
    2. Compute VADER sentiment scores on raw review text
    3. Adjust LSTM prediction based on VADER scores
    4. Return both original LSTM and adjusted ratings for transparency
    
    Args:
        review_text (str): The product review text
        model: Trained LSTM model
        tokenizer: Keras tokenizer fitted on training data
        label_encoder: Scikit-learn label encoder for rating classes
    
    Returns:
        dict: Contains:
            - lstm_rating: Rating directly from LSTM model
            - lstm_confidence: LSTM confidence percentage
            - sentiment_scores: VADER sentiment analysis result
            - sentiment_category: "Negative", "Neutral", or "Positive"
            - final_rating: Adjusted rating after sentiment correction
            - adjustment: Adjustment amount
            - adjustment_reason: Why rating was adjusted
            - probabilities: LSTM probability distribution
    """
    # Get LSTM prediction
    processed_text = preprocess_text(review_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    
    # Use max_length=250 to match training pipeline
    max_length = 250
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    lstm_rating = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get probability distribution
    all_classes = label_encoder.classes_
    probabilities = {str(rating): float(prob) * 100 for rating, prob in zip(all_classes, prediction[0])}
    
    # Get VADER sentiment scores
    sentiment_scores = compute_sentiment_score(review_text)
    
    # Adjust rating based on sentiment
    adjusted_rating, adjustment, reason = adjust_rating_with_sentiment(lstm_rating, sentiment_scores)
    
    # Get sentiment category
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


# Main Streamlit App
def main():
    """Main Streamlit application with improved UI and preprocessing."""
    
    # Set page configuration
    st.set_page_config(
        page_title="Review Rating Predictor",
        page_icon="⭐",
        layout="wide"
    )
    
    # Sidebar with model information
    with st.sidebar:
        st.title("📚 About This App")
        
        st.markdown("""
        ### How It Works
        This Review Rating Predictor uses a trained **LSTM neural network** to analyze 
        product reviews and predict their ratings (1-5 stars).
        
        ### Dataset
        - **Source**: Product review dataset
        - **Reviews**: Thousands of customer reviews
        - **Model**: LSTM (Long Short-Term Memory) network
        - **Training**: Deep learning with TensorFlow/Keras
        
        ### Key Improvements
        **Smart Preprocessing**: 
        - Preserves negation words ("not", "never", "no")
        - Preserves sentiment words ("bad", "good", "amazing")
        - Removes only generic stopwords
        
        **Better Predictions**:
        - Handles sentiment reversals correctly
        - Improves accuracy on negated reviews
        - Works with short reviews
        
        ### Negation Words Preserved
        not, no, never, nor, don't, didn't, won't, can't, shouldn't
        
        ### Example Sentiment Words Preserved
        bad, awful, terrible, excellent, amazing, great, fantastic
        """)
        
        st.divider()
        st.markdown("**Created with Streamlit & TensorFlow**")
    
    # Main content
    st.title("⭐ Review Rating Predictor")
    
    st.markdown("""
    Enter a product review and our LSTM model will predict the rating. 
    The model analyzes the sentiment and content of your review to determine the likely rating.
    """)
    
    # Load model and artifacts
    model, tokenizer, label_encoder = load_model_and_artifacts()
    
    # Example reviews section
    st.markdown("### 💡 Try These Examples")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    examples = {
        "Negative": "not good, never buying again",
        "Positive": "amazing product, excellent quality",
        "Mixed": "good quality but bad customer service"
    }
    
    example_reviews = {
        "Negative": examples["Negative"],
        "Positive": examples["Positive"],
        "Mixed": examples["Mixed"]
    }
    
    # Store selected example in session state
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    
    with example_col1:
        if st.button("📉 Negative Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Negative"]
            st.rerun()
    
    with example_col2:
        if st.button("📈 Positive Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Positive"]
            st.rerun()
    
    with example_col3:
        if st.button("↔️ Mixed Example", use_container_width=True):
            st.session_state.review_text = example_reviews["Mixed"]
            st.rerun()
    
    st.divider()
    
    # Create text area for user input
    review_input = st.text_area(
        label="Enter your product review:",
        placeholder="Example: This product is amazing! It works perfectly and exceeded my expectations.",
        height=150,
        value=st.session_state.review_text,
        key="review_input"
    )
    
    # Create prediction button
    if st.button("🔮 Predict Rating", use_container_width=True):
        # Handle empty input
        if not review_input.strip():
            st.warning("⚠️ Please enter a review text before predicting.")
        else:
            # Update session state
            st.session_state.review_text = review_input
            
            # Show loading message
            with st.spinner("Analyzing review..."):
                try:
                    # Get hybrid prediction (LSTM + VADER)
                    result = predict_rating(
                        review_input, 
                        model, 
                        tokenizer, 
                        label_encoder
                    )
                    
                    # Extract results from dictionary
                    lstm_rating = result['lstm_rating']
                    lstm_confidence = result['lstm_confidence']
                    sentiment_scores = result['sentiment_scores']
                    sentiment_category = result['sentiment_category']
                    final_rating = result['final_rating']
                    adjustment = result['adjustment']
                    adjustment_reason = result['adjustment_reason']
                    probabilities = result['probabilities']
                    processed_text = result['processed_text']
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Create columns for main prediction results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🤖 LSTM Prediction",
                            value=f"{lstm_rating} ⭐",
                            delta=f"{lstm_confidence:.1f}% confidence"
                        )
                    
                    with col2:
                        # Show color-coded sentiment
                        if sentiment_category == "Negative":
                            sentiment_emoji = "😞"
                            sentiment_color = "red"
                        elif sentiment_category == "Positive":
                            sentiment_emoji = "😊"
                            sentiment_color = "green"
                        else:
                            sentiment_emoji = "😐"
                            sentiment_color = "blue"
                        
                        st.metric(
                            label="💭 Sentiment",
                            value=f"{sentiment_emoji} {sentiment_category}",
                            delta=f"Score: {sentiment_scores['compound']:.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="✨ Final Rating",
                            value=f"{final_rating} ⭐",
                        )
                    
                    st.divider()
                    
                    # Show sentiment adjustment details
                    st.subheader("🔧 Sentiment-Based Adjustment")
                    
                    if adjustment != 0:
                        adjustment_text = f"{adjustment:+d}" if adjustment > 0 else f"{adjustment}"
                        st.info(
                            f"**Adjustment Applied:** {adjustment_text} star(s)\n\n"
                            f"**Reason:** {adjustment_reason}"
                        )
                    else:
                        st.info(
                            f"**No Adjustment:** LSTM prediction aligns with sentiment\n\n"
                            f"{adjustment_reason}"
                        )
                    
                    # Show sentiment meter
                    st.subheader("📊 Sentiment Analysis Breakdown")
                    
                    sentiment_col1, sentiment_col2 = st.columns(2)
                    
                    with sentiment_col1:
                        st.metric(
                            label="Negative Sentiment",
                            value=f"{sentiment_scores['neg']*100:.1f}%"
                        )
                        st.metric(
                            label="Positive Sentiment",
                            value=f"{sentiment_scores['pos']*100:.1f}%"
                        )
                    
                    with sentiment_col2:
                        st.metric(
                            label="Neutral Sentiment",
                            value=f"{sentiment_scores['neu']*100:.1f}%"
                        )
                        st.metric(
                            label="Compound Score",
                            value=f"{sentiment_scores['compound']:.3f}",
                            delta="(-1 to +1 scale)"
                        )
                    
                    st.divider()
                    
                    # FEATURE 1: Sentiment Gauge Visualization
                    # This gauge provides an intuitive visual representation of the VADER compound score
                    # helping users understand the linguistic sentiment independent of LSTM predictions
                    st.subheader("🎯 Sentiment Intensity Gauge")
                    st.markdown("""
                    This gauge visualizes the VADER sentiment compound score on a scale from -1 (very negative) 
                    to +1 (very positive). A neutral sentiment is at 0. The gauge color changes to reflect the 
                    sentiment polarity, helping you quickly understand the emotional tone of the review.
                    """)
                    
                    gauge_fig = create_sentiment_gauge(sentiment_scores['compound'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    st.divider()
                    
                    # FEATURE 2: Word Importance Visualization
                    # Highlight sentiment words to improve interpretability and transparency
                    st.subheader("🔑 Key Sentiment Words Detected")
                    st.markdown("""
                    The words highlighted below are sentiment indicators detected in your review. 
                    **Red words** indicate negative sentiment, while **green words** indicate positive sentiment.
                    These words directly influence the VADER sentiment score and help explain why the model 
                    adjusted the rating in a particular direction.
                    """)
                    
                    # Get detected sentiment words
                    detected_negative, detected_positive = get_detected_sentiment_words(review_input)
                    
                    # Display detected sentiment words
                    sentiment_metrics_col1, sentiment_metrics_col2 = st.columns(2)
                    
                    with sentiment_metrics_col1:
                        if detected_negative:
                            st.error(f"**Negative Words Detected:** {', '.join(detected_negative)}")
                        else:
                            st.success("**No negative sentiment words detected**")
                    
                    with sentiment_metrics_col2:
                        if detected_positive:
                            st.success(f"**Positive Words Detected:** {', '.join(detected_positive)}")
                        else:
                            st.error("**No positive sentiment words detected**")
                    
                    # Display highlighted review text
                    st.markdown("**Review with Highlighted Sentiment Words:**")
                    highlighted_html = highlight_sentiment_words(review_input)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Show probability distribution
                    st.subheader("📈 Prediction Probability Distribution (LSTM)")
                    
                    # Create bar chart
                    prob_data = pd.DataFrame({
                        'Rating': list(probabilities.keys()),
                        'Probability (%)': list(probabilities.values())
                    })
                    
                    # Sort by rating
                    prob_data['Rating'] = pd.Categorical(prob_data['Rating'], 
                                                         categories=sorted(probabilities.keys()), 
                                                         ordered=True)
                    prob_data = prob_data.sort_values('Rating')
                    
                    st.bar_chart(
                        data=prob_data.set_index('Rating'),
                        height=300,
                        use_container_width=True
                    )
                    
                    # Show detailed probabilities
                    st.markdown("### Rating Probabilities")
                    prob_cols = st.columns(len(probabilities))
                    for idx, (rating, prob) in enumerate(sorted(probabilities.items())):
                        with prob_cols[idx]:
                            st.metric(f"{rating} ⭐", f"{prob:.2f}%")
                    
                    st.divider()
                    
                    # Show the preprocessed text with highlights
                    st.subheader("📋 Text Processing Details")
                    
                    highlighted = highlight_negation_words(processed_text)
                    
                    col_original, col_processed = st.columns(2)
                    
                    with col_original:
                        st.markdown("**Original Text:**")
                        st.text(review_input)
                    
                    with col_processed:
                        st.markdown("**Cleaned Text (used by model):**")
                        st.markdown(highlighted)
                    
                    # Show what was removed
                    with st.expander("🔍 Stopword Removal Details"):
                        original_words = set(review_input.lower().split())
                        cleaned_words = set(processed_text.split())
                        removed_words = original_words - cleaned_words
                        
                        if removed_words:
                            st.markdown("**Words removed during preprocessing:**")
                            st.markdown(", ".join(sorted(removed_words)) if removed_words else "None")
                        else:
                            st.markdown("**No common stopwords were removed** (this is good!)")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    
    # Add footer with explanation
    st.divider()
    st.markdown("""
    ---
    ### 🎯 Hybrid Prediction Approach: LSTM + VADER Sentiment Analysis
    
    This app uses a **two-stage prediction system** to improve accuracy:
    
    **Stage 1: Deep Learning (LSTM Neural Network)**
    - Learns complex patterns from training data
    - Captures subtle sentiment nuances
    - Generates initial rating prediction
    
    **Stage 2: Rule-Based Correction (VADER Sentiment Analysis)**
    - Analyzes sentiment polarity of the review text
    - Detects strong negative/positive signals
    - Adjusts LSTM prediction if sentiment contradicts it
    
    **Why This Hybrid Approach Works:**
    
    Training data often contains **noisy labels** - reviews marked with the wrong rating.
    For example, some 5-star reviews contain very negative language. A pure LSTM model 
    would learn to ignore the negative language and predict 5 stars anyway.
    
    By combining LSTM predictions with VADER sentiment analysis, we:
    ✅ Correct obvious errors from noisy training data
    ✅ Improve performance on negation-heavy reviews ("not good", "never again")
    ✅ Maintain transparency by showing both the LSTM and adjusted predictions
    ✅ Provide linguistic explanations for rating adjustments
    
    **Adjustment Logic:**
    - Very negative (compound < -0.5) → Reduce rating by 2 stars
    - Negative (-0.5 to -0.1) → Reduce rating by 1 star
    - Neutral (-0.1 to 0.1) → No adjustment
    - Very positive (compound > 0.5) → Increase rating by 1 star
    
    **Why Negation and Sentiment Words Matter:**
    
    - **"not good"** → Without "not": "good" → LSTM predicts positive ❌
    - **"not good"** → With "not": Sentiment = -0.45 → Reduces rating ✅
    
    This app **preserves critical sentiment words** to maintain accuracy:
    - Negation words (not, never, no, don't) reverse sentiment polarity
    - Sentiment words (bad, good, amazing) carry the core emotion
    
    **Result**: Better predictions on negated and sentiment-heavy reviews! 🎯
    """)

if __name__ == "__main__":
    main()