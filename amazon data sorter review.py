import os
import re
import random
import numpy as np
import pandas as pd
import gzip
import csv
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

SPELL = SpellChecker()

# --------- Text cleaning ----------
def clean_text(text, do_spell=True):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)               # Remove HTML tags
    text = re.sub(r"http\S+", " ", text)             # Remove URLs
    text = re.sub(r"[^\w\s]", " ", text)             # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces
    
    if do_spell:
        corrected_words = []
        for word in text.split():
            if word in SPELL:
                corrected_words.append(SPELL.correction(word))
            else:
                corrected_words.append(word)
        text = " ".join(corrected_words)
    
    return text

# --------- Load positive, negative, and unlabeled reviews ----------
def load_amazon_reviews(root_folder):
    reviews = []
    labels = []
    categories = []

    for category in os.listdir(root_folder):
        cat_path = os.path.join(root_folder, category)
        if not os.path.isdir(cat_path):
            continue
        
        for file in os.listdir(cat_path):
            file_path = os.path.join(cat_path, file)

            # Label handling
            if file == "positive.review":
                label = 1
            elif file == "negative.review":
                label = 0
            elif file == "unlabeled.review":
                label = "unlabeled"
            elif file.endswith(".txt") and "unlabeled" in file:
                label = "unlabeled"
            else:
                continue  # Skip any other files

            # Read file
            try:
                if file.endswith(".gz"):
                    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                continue

            # Process lines, skip empties, clean text
            for line in lines:
                line = line.strip()
                if len(line.split()) < 3:  # STRICT removal of short / meaningless lines
                    continue

                cleaned = clean_text(line)
                if len(cleaned.split()) < 3:  # Just in case cleaning made it too short
                    continue

                reviews.append(cleaned)
                labels.append(label)
                categories.append(category)

    print(f"Total valid reviews loaded: {len(reviews)} ✅")
    return reviews, labels, categories

# --------- Remove outliers ----------
def remove_outliers(reviews, labels, categories, min_len=4, max_len=300):
    filtered_r, filtered_l, filtered_c, filtered_cat = [], [], [], []
    for r, l, c in zip(reviews, labels, categories):
        length = len(r.split())
        if length < min_len or length > max_len:
            continue
        filtered_r.append(r)
        filtered_l.append(l)
        filtered_cat.append(c)
    print(f"Remaining after outlier removal: {len(filtered_r)} ✅")
    return filtered_r, filtered_l, filtered_cat

# --------- Train model and split data ----------
def train_model_pipeline(reviews, labels, max_words=10000, max_len=200):
    # Keep only labeled positive/negative for training
    train_reviews = []
    train_labels = []
    for r, l in zip(reviews, labels):
        if l == "unlabeled":
            continue
        train_reviews.append(r)
        train_labels.append(l)

    # Shuffle
    combined = list(zip(train_reviews, train_labels))
    random.shuffle(combined)
    train_reviews, train_labels = zip(*combined)

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_reviews)
    sequences = tokenizer.texts_to_sequences(train_reviews)

    # Pad/truncate
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Split into Train/Validation/Test (80/10/10)
    x_train, x_temp, y_train, y_temp = train_test_split(
        padded, train_labels, train_size=0.8, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # Convert labels to numpy
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    print("Dataset split into training, validation, test ✅")

    # Build Model
    model = Sequential([
        Embedding(max_words, 64, input_shape=(max_len,)),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    print("Training model 🧠...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), batch_size=32)
    print("Training complete ✅")

    # Evaluate
    print("Testing on unseen data 🔍")
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}% ✅\n")

    return model, tokenizer

# --------- Predict ALL reviews and save separate CSVs ----------
def predict_all_and_save_csv(model, tokenizer, reviews, labels, categories, root_folder):
    max_len = 200
    all_data_by_category = {}

    for r, l, cat in zip(reviews, labels, categories):
        if cat not in all_data_by_category:
            all_data_by_category[cat] = []
        all_data_by_category[cat].append((r, cat))

    # Predict per category
    for cat, items in all_data_by_category.items():
        texts = [i[0] for i in items]
        cleaned = [t for t in texts]  # Already cleaned earlier
        seq = tokenizer.texts_to_sequences(cleaned)
        pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

        preds = model.predict(pad, batch_size=32)

        rows = []
        for text, score in zip(cleaned, preds):
            score = float(score[0])
            label = "positive" if score >= 0.5 else "negative"
            rows.append([cat, text, label, score])

        df = pd.DataFrame(rows, columns=["category", "review", "predicted_label", "score"])

        output_file = os.path.join(root_folder, f"{cat}_predictions.csv")
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

        print(f"Predictions saved for '{cat}' → {output_file} ✅")

    print("\nAll category CSV files generated 🎉")

# ------------------- Run Everything -------------------
if __name__ == "__main__":
    dataset_base = r"C:\Users\CCC\Downloads\amazon_reviews\domain_sentiment_data\sorted_data_acl"
    
    # 1. Load reviews
    rev, lab, cat = load_amazon_reviews(dataset_base)
    # 2. Remove outliers
    rev, lab, cat = remove_outliers(rev, lab, cat)

    # 3. Train model
    model, tokenizer = train_model_pipeline(rev, lab)

    # 4. Predict ALL reviews and save CSV per category
    predict_all_and_save_csv(model, tokenizer, rev, lab, cat, dataset_base)
