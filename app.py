import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import base64
import io
import zipfile
from collections import Counter

# Text preprocessing imports
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Enhanced explainability with LIME/SHAP
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

# Enhanced preprocessing function
@st.cache_data
def enhanced_preprocessing(text):
    """Enhanced text preprocessing function with error handling."""
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
        st.error(f"Preprocessing error: {e}")
        return ""

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the latest trained model and vectorizer."""
    try:
        # Find the latest enhanced model files
        model_files = glob.glob('models/enhanced/best_fake_news_model_enhanced_*.joblib')
        vectorizer_files = glob.glob('models/enhanced/tfidf_vectorizer_enhanced_*.joblib')
        
        if not model_files or not vectorizer_files:
            # Fallback to regular models
            model_files = glob.glob('models/best_fake_news_model.joblib')
            vectorizer_files = glob.glob('models/tfidf_vectorizer.joblib')
        
        if not model_files or not vectorizer_files:
            return None, None, "No model files found"
        
        # Load the latest files
        latest_model = max(model_files, key=os.path.getctime)
        latest_vectorizer = max(vectorizer_files, key=os.path.getctime)
        
        model = joblib.load(latest_model)
        vectorizer = joblib.load(latest_vectorizer)
        
        return model, vectorizer, f"Loaded: {os.path.basename(latest_model)}"
    
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Prediction function
def predict_fake_news(text, model, vectorizer):
    """Enhanced production-ready fake news prediction function."""
    try:
        # Preprocess text
        processed_text = enhanced_preprocessing(text)
        
        if not processed_text:
            return {
                'error': 'Text preprocessing resulted in empty content',
                'original_text': text[:100] + '...' if len(text) > 100 else text
            }
        
        # Transform text
        text_features = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        prediction_proba = model.predict_proba(text_features)[0]
        
        # Prepare results
        result = {
            'prediction': {
                'label': 'FAKE' if prediction == 0 else 'REAL',
                'numeric_label': int(prediction),
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    'fake': float(prediction_proba[0]),
                    'real': float(prediction_proba[1])
                }
            },
            'text_analysis': {
                'original_length': len(text),
                'processed_length': len(processed_text),
                'word_count': len(processed_text.split()) if processed_text else 0
            },
            'details': {
                'original_text': text,
                'processed_text': processed_text
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'original_text': text[:100] + '...' if len(text) > 100 else text
        }

# Streamlit App
def main():
    st.set_page_config(
        page_title="🔍 Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fake News Detector")
    st.markdown("### AI-Powered News Article Authenticity Checker")
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK data..."):
        nltk_status = download_nltk_data()
    
    if not nltk_status:
        st.error("Failed to download NLTK data. Some features may not work.")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, vectorizer, status_msg = load_model()
    
    st.sidebar.success(status_msg)
    
    if model is None:
        st.error("❌ Could not load the model. Please ensure model files exist in the models/ directory.")
        st.info("Run the notebook first to train and save the model.")
        return
    
    # Model info
    st.sidebar.info(f"**Model Type:** {type(model).__name__}")
    if hasattr(vectorizer, 'vocabulary_'):
        st.sidebar.info(f"**Vocabulary Size:** {len(vectorizer.vocabulary_):,}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📰 Enter News Article")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
        
        if input_method == "Paste Text":
            text_input = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter the news article text you want to analyze for authenticity..."
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            text_input = ""
            if uploaded_file is not None:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded content:", text_input, height=200, disabled=True)
        
        # Predict button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing article..."):
                    result = predict_fake_news(text_input, model, vectorizer)
                
                # Display results
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    pred = result['prediction']
                    
                    # Result display
                    if pred['label'] == 'FAKE':
                        st.error(f"🚨 **FAKE NEWS DETECTED**")
                        st.error(f"Confidence: {pred['confidence']:.1%}")
                    else:
                        st.success(f"✅ **APPEARS AUTHENTIC**")
                        st.success(f"Confidence: {pred['confidence']:.1%}")
                    
                    # Detailed results
                    with st.expander("📊 Detailed Analysis"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Fake Probability", f"{pred['probabilities']['fake']:.1%}")
                            st.metric("Original Length", f"{result['text_analysis']['original_length']:,} chars")
                        
                        with col_b:
                            st.metric("Real Probability", f"{pred['probabilities']['real']:.1%}")
                            st.metric("Word Count", f"{result['text_analysis']['word_count']:,} words")
                        
                        # Probability chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        labels = ['Fake', 'Real']
                        probs = [pred['probabilities']['fake'], pred['probabilities']['real']]
                        colors = ['#ff6b6b', '#4ecdc4']
                        
                        bars = ax.bar(labels, probs, color=colors, alpha=0.8)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        ax.set_ylim(0, 1)
                        
                        # Add value labels on bars
                        for bar, prob in zip(bars, probs):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    # Processed text comparison
                    with st.expander("🔧 Text Processing Details"):
                        st.subheader("Original Text:")
                        st.text(result['details']['original_text'][:500] + "..." if len(result['details']['original_text']) > 500 else result['details']['original_text'])
                        
                        st.subheader("Processed Text:")
                        st.text(result['details']['processed_text'][:500] + "..." if len(result['details']['processed_text']) > 500 else result['details']['processed_text'])
            
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.header("ℹ️ How It Works")
        
        st.markdown("""
        **Our AI Model:**
        - Trained on 12,000+ news articles
        - Uses machine learning classification
        - Analyzes text patterns and language
        
        **Features:**
        - Real-time analysis
        - Confidence scoring
        - Text preprocessing
        - Feature extraction
        
        **Accuracy:**
        - High precision detection
        - Balanced for both fake and real news
        - Handles various article types
        """)
        
        st.header("⚠️ Important Notes")
        st.warning("""
        - This tool provides AI-based suggestions
        - Always verify with multiple sources
        - Consider context and publication date
        - Use as part of broader fact-checking
        """)
        
        st.header("📈 Sample Articles")
        
        sample_fake = """BREAKING: Scientists discover aliens in Area 51! Government tries to hide the truth but whistleblower reveals shocking details. You won't believe what they found!"""
        
        sample_real = """The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting, citing concerns about inflation and economic stability in the current market conditions."""
        
        if st.button("Try Sample Fake News"):
            st.session_state.sample_text = sample_fake
        
        if st.button("Try Sample Real News"):
            st.session_state.sample_text = sample_real
        
        # Auto-fill sample text
        if 'sample_text' in st.session_state:
            text_input = st.session_state.sample_text
            del st.session_state.sample_text
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("**🔬 Built with Machine Learning | 📊 Powered by Streamlit**")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
