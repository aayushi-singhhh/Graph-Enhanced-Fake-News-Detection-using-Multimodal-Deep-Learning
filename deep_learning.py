import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, Bidirectional, Embedding, Dropout, 
        GlobalMaxPooling1D, Conv1D, MaxPooling1D
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

# Traditional ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Text preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class DeepLearningFakeNewsDetector:
    """
    Enhanced fake news detector with deep learning models.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.tokenizer = None
        self.max_features = 10000
        self.max_length = 500
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Set random seeds
        np.random.seed(random_state)
        if TF_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            print("⚠️ Warning: Could not download NLTK data")
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing."""
        if pd.isna(text) or text == '':
            return ""
        
        try:
            # Initialize preprocessing tools
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            # Preprocessing steps
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenization and lemmatization
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens 
                      if word not in stop_words and len(word) > 2]
            
            return ' '.join(tokens)
        
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return ""
    
    def load_data(self, data_path=None):
        """Load and preprocess data."""
        try:
            # Try to load existing processed data
            if os.path.exists('../data/df_final.pickle'):
                print("Loading existing processed data...")
                with open('../data/df_final.pickle', 'rb') as f:
                    df_final = pickle.load(f)
            else:
                print("Processing raw data...")
                # Load and process raw data (implement based on your data structure)
                df = pd.read_csv('../data/fake.csv')
                
                # Basic preprocessing (adapt based on your data structure)
                df_clean = df[['title', 'text', 'type']].copy()
                df_clean = df_clean.dropna()
                df_clean['content'] = df_clean['title'].astype(str) + ' ' + df_clean['text'].astype(str)
                
                # Create binary labels
                fake_categories = ['fake', 'bias', 'conspiracy', 'hate', 'junksci']
                real_categories = ['bs', 'satire', 'state']
                
                def map_to_binary(type_val):
                    if type_val in fake_categories:
                        return 0  # fake
                    elif type_val in real_categories:
                        return 1  # real
                    else:
                        return -1  # unknown
                
                df_clean['label'] = df_clean['type'].apply(map_to_binary)
                df_clean = df_clean[df_clean['label'] != -1]
                
                # Preprocess text
                df_clean['clean_content'] = df_clean['content'].apply(self.preprocess_text)
                df_final = df_clean[['clean_content', 'label']].copy()
                
                # Save processed data
                os.makedirs('../data', exist_ok=True)
                with open('../data/df_final.pickle', 'wb') as f:
                    pickle.dump(df_final, f)
            
            print(f"✅ Data loaded: {len(df_final)} samples")
            print(f"   - Fake: {(df_final['label'] == 0).sum()}")
            print(f"   - Real: {(df_final['label'] == 1).sum()}")
            
            return df_final
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def prepare_sequences(self, texts, labels=None, fit_tokenizer=True):
        """Prepare text sequences for deep learning models."""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None, None
        
        try:
            # Create tokenizer
            if fit_tokenizer or self.tokenizer is None:
                self.tokenizer = Tokenizer(
                    num_words=self.max_features,
                    oov_token='<OOV>',
                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
                )
                self.tokenizer.fit_on_texts(texts)
                print(f"✅ Tokenizer fitted on {len(texts)} texts")
                print(f"   - Vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Convert texts to sequences
            sequences = self.tokenizer.texts_to_sequences(texts)
            
            # Pad sequences
            X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            
            print(f"✅ Sequences prepared: {X.shape}")
            
            if labels is not None:
                y = np.array(labels)
                return X, y
            else:
                return X, None
            
        except Exception as e:
            print(f"❌ Error preparing sequences: {str(e)}")
            return None, None
    
    def create_lstm_model(self, embedding_dim=100, lstm_units=64):
        """Create LSTM model for fake news detection."""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        try:
            model = Sequential([
                Embedding(self.max_features, embedding_dim, input_length=self.max_length),
                Dropout(0.3),
                LSTM(lstm_units, return_sequences=True),
                Dropout(0.3),
                LSTM(lstm_units//2),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ LSTM model created")
            return model
            
        except Exception as e:
            print(f"❌ Error creating LSTM model: {str(e)}")
            return None
    
    def create_bilstm_model(self, embedding_dim=100, lstm_units=64):
        """Create Bidirectional LSTM model."""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        try:
            model = Sequential([
                Embedding(self.max_features, embedding_dim, input_length=self.max_length),
                Dropout(0.3),
                Bidirectional(LSTM(lstm_units, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(lstm_units//2)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ BiLSTM model created")
            return model
            
        except Exception as e:
            print(f"❌ Error creating BiLSTM model: {str(e)}")
            return None
    
    def create_cnn_model(self, embedding_dim=100, filters=64, kernel_size=3):
        """Create CNN model for text classification."""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        try:
            model = Sequential([
                Embedding(self.max_features, embedding_dim, input_length=self.max_length),
                Dropout(0.3),
                Conv1D(filters, kernel_size, activation='relu'),
                MaxPooling1D(pool_size=2),
                Conv1D(filters//2, kernel_size, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ CNN model created")
            return model
            
        except Exception as e:
            print(f"❌ Error creating CNN model: {str(e)}")
            return None
    
    def train_deep_model(self, model, X_train, y_train, X_val, y_val, 
                        model_name="DeepModel", epochs=10, batch_size=32):
        """Train deep learning model with callbacks."""
        if not TF_AVAILABLE or model is None:
            print("❌ TensorFlow not available or model is None")
            return None
        
        try:
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
            
            print(f"🔄 Training {model_name}...")
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            print(f"✅ {model_name} training completed")
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'history': history,
                'type': 'deep_learning'
            }
            
            return history
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            return None
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model combining multiple algorithms."""
        try:
            print("🔄 Creating ensemble model...")
            
            # Base models
            models = []
            
            # Logistic Regression
            lr_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            models.append(('lr', lr_model))
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            models.append(('rf', rf_model))
            
            # XGBoost (if available)
            if XGB_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
                models.append(('xgb', xgb_model))
            
            # Create voting ensemble
            ensemble = VotingClassifier(
                estimators=models,
                voting='soft'  # Use predicted probabilities
            )
            
            print(f"✅ Ensemble model created with {len(models)} base models")
            return ensemble
            
        except Exception as e:
            print(f"❌ Error creating ensemble model: {str(e)}")
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model", model_type="traditional"):
        """Evaluate model performance."""
        try:
            if model_type == "deep_learning":
                # Deep learning model evaluation
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                # Traditional ML model evaluation
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            print(f"📊 {model_name} Performance:")
            for metric, value in metrics.items():
                print(f"   - {metric.capitalize()}: {value:.4f}")
            
            return metrics, y_pred, y_pred_proba
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            return None, None, None
    
    def train_all_models(self, df):
        """Train all available models."""
        print("🚀 Training All Deep Learning Models")
        print("=" * 60)
        
        # Prepare data
        X_text = df['clean_content']
        y = df['label']
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"📊 Data split completed:")
        print(f"   - Training: {len(X_train_text)} samples")
        print(f"   - Testing: {len(X_test_text)} samples")
        
        results = {}
        
        # Traditional ML models (for comparison and ensemble)
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print(f"\n1️⃣ Preparing TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)
        
        # Ensemble model
        print(f"\n2️⃣ Training Ensemble Model...")
        ensemble_model = self.create_ensemble_model(X_train_tfidf, y_train)
        if ensemble_model:
            ensemble_model.fit(X_train_tfidf, y_train)
            metrics, _, _ = self.evaluate_model(ensemble_model, X_test_tfidf, y_test, "Ensemble")
            if metrics:
                results["Ensemble"] = metrics
        
        # Deep Learning models
        if TF_AVAILABLE:
            print(f"\n3️⃣ Preparing sequences for deep learning...")
            
            # Prepare sequences
            X_train_seq, y_train_seq = self.prepare_sequences(X_train_text, y_train, fit_tokenizer=True)
            X_test_seq, y_test_seq = self.prepare_sequences(X_test_text, y_test, fit_tokenizer=False)
            
            if X_train_seq is not None:
                # Split train into train/validation
                X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
                    X_train_seq, y_train_seq, test_size=0.2, random_state=self.random_state, stratify=y_train_seq
                )
                
                # LSTM Model
                print(f"\n4️⃣ Training LSTM Model...")
                lstm_model = self.create_lstm_model()
                if lstm_model:
                    self.train_deep_model(lstm_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, 
                                        "LSTM", epochs=10, batch_size=32)
                    metrics, _, _ = self.evaluate_model(lstm_model, X_test_seq, y_test_seq, "LSTM", "deep_learning")
                    if metrics:
                        results["LSTM"] = metrics
                
                # BiLSTM Model
                print(f"\n5️⃣ Training BiLSTM Model...")
                bilstm_model = self.create_bilstm_model()
                if bilstm_model:
                    self.train_deep_model(bilstm_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, 
                                        "BiLSTM", epochs=10, batch_size=32)
                    metrics, _, _ = self.evaluate_model(bilstm_model, X_test_seq, y_test_seq, "BiLSTM", "deep_learning")
                    if metrics:
                        results["BiLSTM"] = metrics
                
                # CNN Model
                print(f"\n6️⃣ Training CNN Model...")
                cnn_model = self.create_cnn_model()
                if cnn_model:
                    self.train_deep_model(cnn_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, 
                                        "CNN", epochs=10, batch_size=32)
                    metrics, _, _ = self.evaluate_model(cnn_model, X_test_seq, y_test_seq, "CNN", "deep_learning")
                    if metrics:
                        results["CNN"] = metrics
        
        # Results comparison
        if results:
            print(f"\n📊 Model Comparison Results:")
            print("=" * 60)
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.round(4)
            print(comparison_df.to_string())
            
            # Find best model
            best_model = comparison_df['f1'].idxmax()
            best_f1 = comparison_df.loc[best_model, 'f1']
            print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"results/deep_learning_results_{timestamp}.csv"
            os.makedirs("results", exist_ok=True)
            comparison_df.to_csv(results_path)
            print(f"✅ Results saved to: {results_path}")
            
        return results

def main():
    """Main function for deep learning experiments."""
    
    print("🧠 Deep Learning Fake News Detection")
    print("=" * 60)
    
    # Create detector
    detector = DeepLearningFakeNewsDetector()
    
    # Load data
    df = detector.load_data()
    if df is None:
        print("❌ Could not load data")
        return
    
    # Train all models
    results = detector.train_all_models(df)
    
    if results:
        print(f"\n✅ Deep learning experiments completed!")
        print(f"📊 {len(results)} models trained and evaluated")
    else:
        print("❌ No models were successfully trained")

if __name__ == "__main__":
    main()
