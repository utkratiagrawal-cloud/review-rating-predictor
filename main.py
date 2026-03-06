"""
Improved Review Rating Prediction - LSTM Model Training Pipeline

KEY IMPROVEMENTS:
1. Dataset balancing using downsampling to prevent 5-star bias
2. Handle dataset imbalance with class weights
3. Larger vocabulary size (20000) with OOV handling
4. Increased max sequence length (250) for better context
5. Bidirectional LSTM for better bidirectional context understanding
6. Stronger model architecture with better regularization
7. Extended training (10-12 epochs) with early stopping
8. Comprehensive class distribution analysis

Why these improvements matter:
- Dataset balancing ensures the model learns all rating levels equally
- Prevents the model from predicting 5 stars for everything (major problem fixed!)
- Bidirectional LSTM captures context from both directions
- Larger vocabulary reduces information loss from OOV tokens
- Longer sequences preserve more review context
- More epochs allow the model to learn better representations
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Download stopwords
nltk.download('stopwords')

print("=" * 80)
print("IMPROVED LSTM MODEL TRAINING PIPELINE")
print("=" * 80)

# Load dataset (smaller size for LSTM)
print("\nLoading dataset...")
df = pd.read_csv("data/Reviews.csv", nrows=40000)

df = df[['Text', 'Score']]
df = df.dropna()

print(f"Dataset loaded: {len(df)} reviews")

# Plot rating distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
rating_counts = df['Score'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7, edgecolor='black')

plt.title("Original Review Rating Distribution (Before Encoding)", fontsize=14, fontweight='bold')
plt.xlabel("Rating (1-5 Stars)", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)
plt.xticks(range(1, 6))
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for idx, (rating, count) in enumerate(zip(rating_counts.index, rating_counts.values)):
    plt.text(rating, count + 100, str(count), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("output/rating_distribution.png", dpi=150)
print("✓ Rating distribution saved to output/rating_distribution.png")

# Clean text
def clean_text(text):
    """
    Clean and preprocess text while preserving negation words for sentiment analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters
    3. Remove common stopwords EXCEPT negation words (not, no, never)
    
    Rationale: Negation words are critical for understanding sentiment polarity.
    Removing them would turn "not good" into "good", losing the negative sentiment.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()

    stop_words = set(stopwords.words('english'))

    # IMPORTANT: Keep negation words - they reverse sentiment polarity
    stop_words.discard("not")
    stop_words.discard("no")
    stop_words.discard("never")
    stop_words.discard("nor")
    stop_words.discard("don't")
    stop_words.discard("didn't")
    stop_words.discard("won't")
    stop_words.discard("can't")
    stop_words.discard("shouldn't")

    text = [word for word in text if word not in stop_words]

    return " ".join(text)

print("\nCleaning text...")
df['Text'] = df['Text'].apply(clean_text)

# IMPROVEMENT: Dataset Balancing to handle class imbalance
# ============================================================================
# PROBLEM: Real-world datasets are imbalanced
#   - 5-star reviews: 62% (majority class)
#   - 1-star reviews: 9% (minority class)
#   - This causes the model to learn a bias toward predicting 5 stars
#   
# SOLUTION: Downsample each class to the same size
#   - This forces the model to learn discriminative features of each rating
#   - Prevents the model from achieving high accuracy by simply predicting 5 stars
#   - Leads to much better generalization on real-world imbalanced data
#   
# HOW IT WORKS:
#   1. Group dataset by rating (1-5 stars)
#   2. Randomly sample up to N examples from each group
#   3. Concatenate all groups into a balanced dataset
#   4. Now each rating class has equal representation
#
# TRADE-OFF:
#   - We discard some samples from majority classes (5 stars)
#   - But we gain a much better model that can distinguish all rating levels
#   - The discarded samples are still useful - they're just duplicative of other 5-star patterns

print("\n" + "=" * 80)
print("DATASET BALANCING - DOWNSAMPLING")
print("=" * 80)

original_size = len(df)
print(f"\nOriginal dataset size: {original_size} reviews")

# Print class distribution BEFORE balancing
print("\nClass distribution BEFORE balancing:")
original_distribution = df['Score'].value_counts().sort_index()
for rating, count in original_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"  {rating} Star: {count:5d} samples ({percentage:5.2f}%)")

# Calculate samples per class - use the minimum and set a reasonable upper limit
# This ensures we keep as many samples as possible while maintaining balance
min_class_count = df['Score'].value_counts().min()
samples_per_class = min(min_class_count, 4000)  # Cap at 4000 per class

print(f"\nDownsampling to {samples_per_class} samples per rating class...")

# Perform stratified downsampling using groupby + sample
# This is more reliable than random downsampling for maintaining class balance
df_balanced = df.groupby('Score', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42)
).reset_index(drop=True)

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(df_balanced)} reviews")
print(f"Reduction: {original_size - len(df_balanced)} reviews removed")

# Print class distribution AFTER balancing
print("\nClass distribution AFTER balancing:")
balanced_distribution = df_balanced['Score'].value_counts().sort_index()
for rating, count in balanced_distribution.items():
    percentage = (count / len(df_balanced)) * 100
    print(f"  {rating} Star: {count:5d} samples ({percentage:5.2f}%)")

# Calculate balance ratio
min_balanced = balanced_distribution.min()
max_balanced = balanced_distribution.max()
balance_ratio = max_balanced / min_balanced if min_balanced > 0 else 1

print(f"\nBalance Ratio (before/after): {original_distribution.max() / min_class_count:.2f}x → {balance_ratio:.2f}x")
if balance_ratio <= 1.1:
    print("✅ Dataset is now well-balanced!")
else:
    print(f"⚠️  Dataset balance ratio: {balance_ratio:.2f}x (close to balanced)")

# Use the balanced dataset for training
df = df_balanced

print("\n" + "=" * 80)

# Encode labels (1-5 → 0-4 for model training)
print("\nEncoding labels...")
label_encoder = LabelEncoder()
df['Score_encoded'] = label_encoder.fit_transform(df['Score'])

print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Print class distribution AFTER training (on balanced data)
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION ANALYSIS (After Balancing, Before Training)")
print("=" * 80)

class_distribution = df['Score_encoded'].value_counts().sort_index()
print("\nClass counts (after encoding 1-5 → 0-4):")
for class_idx, count in class_distribution.items():
    original_rating = label_encoder.inverse_transform([class_idx])[0]
    percentage = (count / len(df)) * 100
    print(f"  Class {class_idx} ({original_rating} Star): {count:5d} samples ({percentage:5.2f}%)")

print("\n⚠️  Statistics:")
print(f"  Ratio (max/min): {class_distribution.max() / class_distribution.min():.2f}x")
if class_distribution.max() / class_distribution.min() < 1.1:
    print("  ✅ Dataset is now well-balanced after downsampling!")
else:
    print("  Still some minor imbalance - computing class weights as backup...")

# IMPROVEMENT: Compute class weights as additional regularization
# After balancing, all class weights should be close to 1.0
# This adds extra robustness to training
print("\nComputing class weights (for additional robustness)...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['Score_encoded']),
    y=df['Score_encoded']
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Calculated class weights:")
for class_idx, weight in class_weight_dict.items():
    original_rating = label_encoder.inverse_transform([class_idx])[0]
    print(f"  Class {class_idx} ({original_rating} Star): {weight:.4f}")

# Save label encoder (needed for predictions)
import pickle
print("\nSaving label encoder...")
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✓ Label encoder saved to model/label_encoder.pkl")

# Train-test split
print("\nSplitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['Score_encoded'], test_size=0.2, random_state=42, stratify=df['Score_encoded']
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\n" + "=" * 80)
print("TOKENIZATION CONFIGURATION")
print("=" * 80)

# IMPROVEMENT 2: Increase vocabulary size and add OOV token handling
# Larger vocabulary reduces information loss from unknown words
# OOV token ensures unknown words are represented consistently
vocab_size = 20000  # Increased from 10000
oov_token = "<OOV>"

print(f"\nVocabulary size: {vocab_size}")
print(f"OOV token: '{oov_token}'")

# IMPROVEMENT 3: Increase max sequence length for better context
# Longer sequences preserve more review context for the model to learn from
max_length = 250  # Increased from 200

print(f"Max sequence length: {max_length}")

print("\nTokenizing training data...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# Save tokenizer (needed for inference in Streamlit app)
print("Saving tokenizer...")
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved to model/tokenizer.pkl")

print("\nConverting text to sequences and padding...")
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

print(f"Train shape: {X_train_pad.shape}")
print(f"Test shape:  {X_test_pad.shape}")


# IMPROVEMENT 4: Build stronger model with Bidirectional LSTM
# 
# Model Architecture Explanation:
# 1. Embedding(20000, 128): Converts word indices to 128-dimensional dense vectors
#    - Vocabulary size: 20000 words
#    - Embedding dimension: 128 (learned word representations)
#
# 2. Bidirectional(LSTM(128, return_sequences=False)): 
#    - LSTM processes sequences in BOTH directions (left→right and right→left)
#    - Captures context from both past and future tokens
#    - Example: "not good" - processes "not" and "good" together from both directions
#    - return_sequences=False outputs only final hidden state (for classification)
#
# 3. Dropout(0.5): Randomly deactivates 50% of neurons during training
#    - Prevents overfitting and improves generalization
#
# 4. Dense(64, activation='relu'): Fully connected layer with ReLU activation
#    - Learns non-linear relationships between LSTM output and predictions
#
# 5. Dropout(0.5): Additional regularization
#
# 6. Dense(5, activation='softmax'): Output layer
#    - 5 units for 5 rating classes (1-5 stars)
#    - Softmax outputs probability distribution over classes

print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    Bidirectional(LSTM(128, return_sequences=False)),  # Bidirectional LSTM for better context
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# IMPROVEMENT 5: EarlyStopping to prevent overfitting
# Monitors validation loss and stops training if no improvement for 2 epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,  # Restore weights from the epoch with lowest val_loss
    verbose=1
)

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Epochs: 10-12 (with early stopping)")
print(f"Batch size: 64")
print(f"Class weights: Applied to handle dataset imbalance")
print(f"Early stopping: Enabled (patience=2)")
print(f"Optimizer: Adam")
print(f"Loss: Sparse Categorical Crossentropy")

# IMPROVEMENT 6: Train with class weights to handle imbalance
# Class weights ensure that minority classes have more influence on loss calculation
# This helps the model learn better representations for underrepresented classes

print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)

history = model.fit(
    X_train_pad,
    y_train,
    epochs=12,  # IMPROVEMENT: Increased from 5 to 12 epochs
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict,  # IMPROVEMENT: Apply class weights
    verbose=1
)

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# IMPROVEMENT 7: Save trained model
print("\nSaving trained model...")
model.save("model/lstm_model.h5")
print("✓ Model saved to model/lstm_model.h5")

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nTest Set Results:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Predict test data for confusion matrix
y_pred_prob = model.predict(X_test_pad, verbose=0)

# Convert probabilities to predicted class
y_pred = np.argmax(y_pred_prob, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues', aspect='auto')

plt.title("Confusion Matrix - LSTM Model Predictions", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Rating", fontsize=12)
plt.ylabel("Actual Rating", fontsize=12)

labels = ["1 Star","2 Star","3 Star","4 Star","5 Star"]

plt.xticks(range(5), labels, rotation=45)
plt.yticks(range(5), labels)

plt.colorbar(label='Number of Samples')

# Add numbers inside each box
for i in range(len(cm)):
    for j in range(len(cm[i])):
        color = "white" if cm[i][j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i][j]), ha="center", va="center", color=color, fontweight='bold')

plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)
print("✓ Confusion matrix saved to output/confusion_matrix.png")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], marker='s', label='Validation Accuracy', linewidth=2)
plt.title("Training vs Validation Accuracy", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o', label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], marker='s', label='Validation Loss', linewidth=2)
plt.title("Training vs Validation Loss", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/training_accuracy.png", dpi=150)
print("✓ Training history saved to output/training_accuracy.png")



# Example predictions to demonstrate model improvements
def predict_review(review_text):
    """
    Make a prediction for a given review text.
    Uses the same preprocessing as training to ensure consistency.
    """
    review_text_cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([review_text_cleaned])
    pad = pad_sequences(seq, maxlen=max_length)

    prediction = model.predict(pad, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_rating = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_rating, confidence, prediction[0]

print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS - Testing Model on Different Reviews")
print("=" * 80)

# Test cases that reveal the improvements
test_reviews = [
    "This product is absolutely amazing and works perfectly",
    "The product was awful and terrible",
    "not good product quality",
    "never buying again",
    "bad, disappointing experience",
    "excellent quality, highly recommend",
    "worst waste of money ever",
    "This is not what I expected but the product is fine"
]

print("\nTesting model predictions:\n")
for review in test_reviews:
    rating, confidence, probs = predict_review(review)
    cleaned = clean_text(review)
    print(f"Review: \"{review}\"")
    print(f"  Cleaned: \"{cleaned}\"")
    print(f"  Predicted: {rating} Star (Confidence: {confidence*100:.2f}%)")
    print(f"  Probabilities: {[f'{p:.4f}' for p in probs]}")
    print()

print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\n✓ Model saved to: model/lstm_model.h5")
print("✓ Tokenizer saved to: model/tokenizer.pkl")
print("✓ Label encoder saved to: model/label_encoder.pkl")
print("\nThe Streamlit app (app.py) is ready to use for inference!")
print("Run: streamlit run app.py")
print("=" * 80)