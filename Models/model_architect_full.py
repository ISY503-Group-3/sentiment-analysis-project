"""
Full conversion of `Model_Architect.ipynb` into a reusable Python module.

This module exposes functions for:
- configuration and environment checks
- loading datasets (supports file paths and file-like objects from Streamlit upload)
- EDA plotting (wordcloud, class balance, review length)
- preprocessing/tokenization (lazy TensorFlow import)
- model builders (BiLSTM, CNN, Transformer) and tournament runner (lazy TF)
- selection and evaluation utilities

TensorFlow imports are deferred until `ensure_tf()` is called to avoid heavy
imports and native runtime logs unless the user explicitly requests training.
"""
from pathlib import Path
import time
import json
import pickle
from typing import Optional, Tuple, Any, List
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wordcloud import WordCloud
    _WC_AVAILABLE = True
except Exception:
    _WC_AVAILABLE = False

# Default hyperparameters (from notebook)
MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS_COMPARE = 6
EPOCHS_FINAL = 20
LEARNING_RATE = 0.0005

# TensorFlow will be imported lazily
tf_available = False


def ensure_tf():
    """Attempt to import TensorFlow and necessary Keras symbols lazily.

    On success, binds symbols to module globals and sets `tf_available=True`.
    Raises ImportError if TF cannot be imported.
    """
    global tf_available
    if tf_available:
        return
    try:
        # Set environment guards before importing TensorFlow to reduce native
        # runtime logging and contention (mutex) issues. These help on many
        # platforms where TF's native runtime prints C++ warnings about locks.
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # INFO=0, WARNING=1, ERROR=2
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('KMP_BLOCKTIME', '1')

        import importlib
        tf = importlib.import_module('tensorflow')
        # Reduce TF Python-side logging
        try:
            import logging
            tf.get_logger().setLevel(logging.ERROR)
        except Exception:
            pass
        # Try to constrain intra/inter op parallelism to avoid thread contention
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            # Some TF builds may not expose these APIs in constrained environments
            pass
        keras = tf.keras
        globals()['tf'] = tf
        globals()['Tokenizer'] = keras.preprocessing.text.Tokenizer
        globals()['pad_sequences'] = keras.preprocessing.sequence.pad_sequences
        globals()['Model'] = keras.models.Model
        globals()['Sequential'] = keras.models.Sequential
        layers = keras.layers
        globals()['Input'] = layers.Input
        globals()['Embedding'] = layers.Embedding
        globals()['Bidirectional'] = layers.Bidirectional
        globals()['LSTM'] = layers.LSTM
        globals()['Dense'] = layers.Dense
        globals()['Dropout'] = layers.Dropout
        globals()['SpatialDropout1D'] = layers.SpatialDropout1D
        globals()['Layer'] = layers.Layer
        globals()['Conv1D'] = layers.Conv1D
        globals()['GlobalMaxPooling1D'] = layers.GlobalMaxPooling1D
        globals()['MultiHeadAttention'] = layers.MultiHeadAttention
        globals()['LayerNormalization'] = layers.LayerNormalization
        globals()['EarlyStopping'] = keras.callbacks.EarlyStopping
        globals()['ModelCheckpoint'] = keras.callbacks.ModelCheckpoint
        globals()['ReduceLROnPlateau'] = keras.callbacks.ReduceLROnPlateau
        globals()['K'] = keras.backend
        tf_available = True
    except Exception:
        tf_available = False
        raise


def load_clean_data(file_paths: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Load and concatenate cleaned CSV files.

    The original notebook loaded four specific CSVs. This function accepts
    a list of file paths (or file-like objects) and concatenates them.
    Returns None if no valid files found.
    """
    if not file_paths:
        return None
    dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    data = pd.concat(dfs, ignore_index=True)
    if 'label' in data.columns:
        data = data[data['label'].isin([0, 1])]
    return data


def load_data(path: Optional[Any] = None, comment_col: str = 'review', score_col: str = 'rating') -> pd.DataFrame:
    """Load a single CSV into a normalized DataFrame with `review` and `label`.

    `path` can be a string/path or a file-like object returned by Streamlit uploader.
    """
    if path is None:
        path = Path(__file__).parent.parent / 'data' / 'input_demo.csv'
    df = pd.read_csv(path)

    # Normalize review
    if 'review' in df.columns:
        df['review'] = df['review'].astype(str)
    elif comment_col in df.columns:
        df['review'] = df[comment_col].astype(str)
    else:
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not text_cols:
            raise ValueError('No text-like column found in CSV')
        df['review'] = df[text_cols[0]].astype(str)

    # Normalize label
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    elif score_col in df.columns:
        s = pd.to_numeric(df[score_col], errors='coerce').fillna(0).astype(int)
        df['label'] = (s >= 4).astype(int) if s.max() > 1 else s
    else:
        # try to locate a binary column
        for c in df.columns:
            vals = set(df[c].dropna().unique())
            if vals.issubset({0, 1}):
                df['label'] = df[c].astype(int)
                break
    if 'label' not in df.columns:
        raise ValueError('No label-like column found in CSV')
    return df[['review', 'label']].copy()


def wordcloud_figure(df: pd.DataFrame, max_words: int = 100):
    text = ' '.join(df['review'].astype(str).values)
    fig, ax = plt.subplots(figsize=(10, 5))
    if _WC_AVAILABLE:
        wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'wordcloud not available; pip install wordcloud', ha='center')
        ax.axis('off')
    return fig


def class_balance_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Compute counts and draw a bar chart to avoid seaborn's future deprecation
    counts = df['label'].value_counts().sort_index()
    # Ensure labels 0 and 1 exist
    counts = counts.reindex([0, 1], fill_value=0)
    colors = sns.color_palette('coolwarm', 2)
    ax.bar([0, 1], counts.values, color=colors)
    ax.set_title('Class Balance')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    return fig


def review_length_figure(df: pd.DataFrame, max_len: int = MAX_LEN):
    df = df.copy()
    df['len'] = df['review'].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['len'], bins=50, kde=True, color='purple', ax=ax)
    ax.axvline(max_len, color='red', linestyle='--', label=f'Truncation Point ({max_len})')
    ax.set_title('Review Length Distribution')
    ax.set_xlabel('Number of Words')
    ax.legend()
    return fig


def preprocess_tokenize(df: pd.DataFrame, max_words: int = MAX_WORDS, max_len: int = MAX_LEN) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Tokenize and pad sequences using Keras Tokenizer. Requires TF via ensure_tf().

    Returns (tokenizer, X_pad, y)
    """
    if not tf_available:
        raise RuntimeError('TensorFlow not available. Call ensure_tf() before preprocessing.')
    X = df['review'].astype(str).values
    y = df['label'].values
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return tokenizer, X_pad, y


def tokenize_split_and_save(df: pd.DataFrame, max_words: int = MAX_WORDS, max_len: int = MAX_LEN,
                            save_artifacts: bool = True, tokenizer_path: str = 'tokenizer.pkl', label_map_path: str = 'label_map.json'):
    """Tokenize, pad, stratified-split (80/10/10) and optionally save artifacts.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test, tokenizer
    """
    tokenizer, X_pad, y = preprocess_tokenize(df, max_words=max_words, max_len=max_len)
    # Stratified splits
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    if save_artifacts:
        try:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            label_map = {0: 'negative', 1: 'positive'}
            with open(label_map_path, 'w') as f:
                json.dump(label_map, f)
        except Exception:
            # ignore artifact save errors — non-critical
            pass

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


# Model builders and training (require TF via ensure_tf())
def _check_tf_available_or_raise():
    if not tf_available:
        raise RuntimeError('TensorFlow not available. Call ensure_tf() before training.')


def build_bilstm(max_len: int = MAX_LEN, max_words: int = MAX_WORDS, embedding_dim: int = EMBEDDING_DIM):
    _check_tf_available_or_raise()
    # Define a small Attention layer inline if not present
    class AttentionLayer(Layer):
        """Custom Attention Layer for Interpretability"""
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
            super(AttentionLayer, self).build(input_shape)
        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

    inputs = Input(shape=(max_len,))
    x = Embedding(max_words, embedding_dim)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='BiLSTM_Attention')


def build_cnn(max_len: int = MAX_LEN, max_words: int = MAX_WORDS, embedding_dim: int = EMBEDDING_DIM):
    _check_tf_available_or_raise()
    inputs = Input(shape=(max_len,))
    x = Embedding(max_words, embedding_dim)(inputs)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='CNN_Baseline')


def build_transformer(max_len: int = MAX_LEN, max_words: int = MAX_WORDS, embedding_dim: int = EMBEDDING_DIM):
    _check_tf_available_or_raise()
    inputs = Input(shape=(max_len,))
    x = Embedding(max_words, embedding_dim)(inputs)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn = Sequential([Dense(64, activation='relu'), Dense(embedding_dim)])
    ffn_output = ffn(x)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    x = GlobalMaxPooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Transformer_Light')


def run_tournament(X_train, y_train, X_val, y_val, epochs: int = 3, batch_size: int = 32):
    _check_tf_available_or_raise()
    models = [build_cnn(), build_transformer(), build_bilstm()]
    results = []
    for m in models:
        start = time.time()
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        h = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
        duration = time.time() - start
        best_val = max(h.history.get('val_accuracy', [0]))
        # store training history dict so callers can visualize per-epoch metrics
        results.append({'Model Name': m.name, 'Val Accuracy': float(best_val), 'Time (s)': duration, 'Model Object': m, 'History': h.history})
    return pd.DataFrame(results)


def plot_training_history(history: dict):
    """Return a matplotlib figure with accuracy and loss curves from a Keras History.history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_acc, ax_loss = axes.ravel()

    # Accuracy
    acc = history.get('accuracy') or history.get('acc')
    val_acc = history.get('val_accuracy') or history.get('val_acc')
    if acc is not None:
        ax_acc.plot(acc, label='train')
    if val_acc is not None:
        ax_acc.plot(val_acc, label='val')
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # Loss
    loss = history.get('loss')
    val_loss = history.get('val_loss')
    if loss is not None:
        ax_loss.plot(loss, label='train')
    if val_loss is not None:
        ax_loss.plot(val_loss, label='val')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()

    plt.tight_layout()
    return fig


def visualize_tournament_results(res_df: pd.DataFrame):
    """Return side-by-side matplotlib figures matching the notebook's visualization.

    Produces (fig_accuracy, fig_time)
    """
    if res_df is None or res_df.empty:
        return None, None
    # Accuracy chart
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    sns.barplot(x='Model Name', y='Val Accuracy', data=res_df, palette='viridis', ax=ax1)
    ax1.set_title('Accuracy Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel('Validation Accuracy')
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    # Time chart
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    sns.barplot(x='Model Name', y='Time (s)', data=res_df, palette='magma', ax=ax2)
    ax2.set_title('Training Speed (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (Seconds)')
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}s', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig1, fig2


def select_final_model(results_df: pd.DataFrame):
    if results_df.empty:
        return None
    sorted_res = results_df.sort_values(by='Val Accuracy', ascending=False).reset_index(drop=True)
    bilstm_row = results_df[results_df['Model Name'] == 'BiLSTM_Attention']
    if not bilstm_row.empty and bilstm_row.iloc[0]['Val Accuracy'] > 0.90:
        return bilstm_row.iloc[0]['Model Object']
    return sorted_res.iloc[0]['Model Object']


def evaluate_model(model, X_test, y_test):
    _check_tf_available_or_raise()
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')

    # ROC plot styled to match notebook visuals
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    try:
        model_name = getattr(model, 'name', 'Model')
    except Exception:
        model_name = 'Model'
    ax_roc.set_title(f'ROC Curve - {model_name}')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate', rotation=45, labelpad=40)
    ax_roc.legend()

    # Confidence histogram (distribution of predicted probabilities)
    fig_conf, ax_conf = plt.subplots(figsize=(6, 4))
    ax_conf.hist(y_pred_prob, bins=25, color='slateblue', alpha=0.85)
    ax_conf.set_title('Model Prediction Confidence Distribution')
    ax_conf.set_xlabel('Predicted Probability')
    ax_conf.set_ylabel('Count')

    return {'report': report, 'confusion_matrix': cm, 'roc_auc': roc_auc, 'fig_cm': fig_cm, 'fig_roc': fig_roc, 'fig_confidence': fig_conf}



def final_train_and_evaluate(final_model, X_train, y_train, X_val, y_val, X_test, y_test,
                             epochs: int = EPOCHS_FINAL, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE):
    """Perform final training on train+val and evaluate on test set.

    Returns a dict with trained model, final_history and evaluation dict (as evaluate_model returns).
    """
    # Ensure TF is available and TF-related symbols are bound in globals
    ensure_tf()
    _check_tf_available_or_raise()

    # Combine train and val for final training
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)

    # Compile with a conservative optimizer
    try:
        final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception:
        # fallback: use tf keras optimizer if available via globals
        tf_g = globals().get('tf')
        if tf_g is not None:
            try:
                final_model.compile(optimizer=tf_g.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
            except Exception:
                final_model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        else:
            final_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # Resolve callback classes from globals (set by ensure_tf)
    RLROP = globals().get('ReduceLROnPlateau')
    ES = globals().get('EarlyStopping')
    callbacks = []
    if RLROP is not None:
        callbacks.append(RLROP(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6))
    if ES is not None:
        callbacks.append(ES(monitor='val_accuracy', patience=3, restore_best_weights=True))

    history = final_model.fit(
        X_final,
        y_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    eval_result = evaluate_model(final_model, X_test, y_test)

    # Save evaluation figures to artifacts for offline inspection and to ensure
    # Streamlit can load them even if rendering the Figure objects fails.
    try:
        # Save into a dedicated artifacts folder inside the project root
        fig_paths = save_evaluation_figs(eval_result, out_dir=str(Path(__file__).parent.parent / 'artifacts'))
        eval_result['fig_paths'] = fig_paths
        # Also write a small JSON summary for easy inspection
        try:
            summary = {
                'roc_auc': eval_result.get('roc_auc'),
                'report': eval_result.get('report'),
                'fig_paths': fig_paths,
            }
            summary_path = Path(__file__).parent.parent / 'artifacts' / 'evaluation_summary.json'
            with open(summary_path, 'w') as sf:
                json.dump(summary, sf, indent=2)
            eval_result['summary_path'] = str(summary_path)
        except Exception:
            pass
    except Exception:
        # non-fatal if saving fails
        pass

    return {'model': final_model, 'history': history, 'evaluation': eval_result}


def save_evaluation_figs(evaluation: dict, out_dir: str = 'artifacts', prefix: str = 'evaluation') -> dict:
    """Save evaluation matplotlib figures to PNG files and return their paths.

    Expects keys `fig_cm`, `fig_roc`, `fig_confidence` in `evaluation`.
    Returns a dict with paths under keys `fig_cm`, `fig_roc`, `fig_confidence`.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    mapping = [('fig_cm', 'confusion_matrix'), ('fig_roc', 'roc_curve'), ('fig_confidence', 'confidence_hist')]
    for key, name in mapping:
        fig = evaluation.get(key)
        if fig is None:
            continue
        try:
            p = Path(out_dir) / f"{prefix}_{name}.png"
            fig.savefig(str(p), bbox_inches='tight')
            paths[key] = str(p)
        except Exception:
            continue
    return paths


def save_artifacts(final_model, tokenizer, label_map, model_path: str = 'final_sentiment_model.keras', tokenizer_path: str = 'tokenizer.pkl', label_map_path: str = 'label_map.json'):
    """Save the final model and artifacts to disk."""
    try:
        final_model.save(model_path)
    except Exception:
        # Some TF builds may not support the .keras format; try HDF5
        try:
            final_model.save(model_path + '.h5')
        except Exception:
            pass

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception:
        pass

    try:
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f)
    except Exception:
        pass


def generate_placeholder_evaluation_figures():
    """Generate placeholder evaluation matplotlib figures and return them.

    This is a TF-free helper used by the Streamlit UI to render example
    Confusion Matrix, ROC curve, and Confidence histogram when real
    evaluation artifacts are not available.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Confusion matrix placeholder
    cm = np.array([[50, 10], [8, 32]])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix (placeholder)')

    # ROC curve placeholder
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(1 - (1 - fpr) ** 2)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = 0.85)')
    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve (placeholder)')
    ax_roc.legend()

    # Confidence histogram placeholder
    probs = np.concatenate([np.random.beta(2, 5, 500), np.random.beta(5, 2, 300)])
    fig_conf, ax_conf = plt.subplots(figsize=(6, 4))
    ax_conf.hist(probs, bins=25, color='slateblue', alpha=0.85)
    ax_conf.set_title('Model Prediction Confidence Distribution (placeholder)')
    ax_conf.set_xlabel('Predicted Probability')
    ax_conf.set_ylabel('Count')

    plt.tight_layout()
    return {'fig_cm': fig_cm, 'fig_roc': fig_roc, 'fig_confidence': fig_conf}
