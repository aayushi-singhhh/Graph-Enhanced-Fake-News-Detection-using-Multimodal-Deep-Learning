"""
Enhanced Streamlit App for Fake News Detection
with batch processing, top features, and report generation
"""

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
import json

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
    st.sidebar.warning("LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.sidebar.warning("SHAP not available. Install with: pip install shap")

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
    # Don't show warning unless user tries to use PDF feature

# Set page config
st.set_page_config(
    page_title="🔍 Enhanced Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .fake-alert {
        background-color: #ffe6e6;
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 1rem;
    }
    .real-alert {
        background-color: #e6ffe6;
        border: 2px solid #44ff44;
        border-radius: 10px;
        padding: 1rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return text

# Load models with caching
@st.cache_resource
def load_models():
    """Load trained models and vectorizer."""
    try:
        model_files = glob.glob('models/**/best_fake_news_model*.joblib', recursive=True)
        if not model_files:
            model_files = glob.glob('models/best_fake_news_model*.joblib')
        
        if model_files:
            model_path = model_files[-1]  # Get latest model
            model = joblib.load(model_path)
            st.sidebar.success(f"✅ Model loaded: {os.path.basename(model_path)}")
        else:
            st.sidebar.error("❌ No model found")
            return None, None
        
        # Load vectorizer
        vectorizer_files = glob.glob('models/**/tfidf_vectorizer*.joblib', recursive=True)
        if not vectorizer_files:
            vectorizer_files = glob.glob('models/tfidf_vectorizer*.joblib')
        
        if vectorizer_files:
            vectorizer_path = vectorizer_files[-1]
            vectorizer = joblib.load(vectorizer_path)
            st.sidebar.success(f"✅ Vectorizer loaded: {os.path.basename(vectorizer_path)}")
        else:
            st.sidebar.error("❌ No vectorizer found")
            return None, None
        
        return model, vectorizer
    
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return None, None

def get_top_features(model, vectorizer, text, n_features=10):
    """Get top influential features for prediction."""
    try:
        # Transform text
        text_vector = vectorizer.transform([enhanced_preprocessing(text)])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get model prediction probabilities
        proba = model.predict_proba(text_vector)[0]
        
        # For tree-based models, try to get feature importance
        if hasattr(model, 'feature_importances_'):
            # Get indices of non-zero features in the text
            nonzero_indices = text_vector.nonzero()[1]
            
            # Get feature importance for these features
            feature_scores = []
            for idx in nonzero_indices:
                feature_name = feature_names[idx]
                importance = model.feature_importances_[idx]
                feature_scores.append((feature_name, importance, text_vector[0, idx]))
            
            # Sort by importance
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            return feature_scores[:n_features]
        
        else:
            # For other models, use coefficient-based importance
            if hasattr(model, 'coef_'):
                coef = model.coef_[0]  # Assuming binary classification
                nonzero_indices = text_vector.nonzero()[1]
                
                feature_scores = []
                for idx in nonzero_indices:
                    feature_name = feature_names[idx]
                    coefficient = coef[idx]
                    tf_idf_score = text_vector[0, idx]
                    importance = abs(coefficient * tf_idf_score)
                    feature_scores.append((feature_name, importance, tf_idf_score))
                
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                return feature_scores[:n_features]
    
    except Exception as e:
        st.error(f"Error getting top features: {e}")
        return []

def predict_single_text(text, model, vectorizer):
    """Make prediction on single text."""
    if not text.strip():
        return None
    
    try:
        # Preprocess
        processed_text = enhanced_preprocessing(text)
        
        # Transform
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get top features
        top_features = get_top_features(model, vectorizer, text)
        
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': max(probabilities),
            'probabilities': {
                'fake': probabilities[0],
                'real': probabilities[1]
            },
            'top_features': top_features,
            'text_length': len(text),
            'processed_length': len(processed_text),
            'word_count': len(text.split())
        }
        
        return result
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def process_batch_texts(texts, model, vectorizer, progress_bar=None):
    """Process multiple texts in batch."""
    results = []
    
    for i, text in enumerate(texts):
        if progress_bar:
            progress_bar.progress((i + 1) / len(texts))
        
        result = predict_single_text(text, model, vectorizer)
        if result:
            results.append(result)
    
    return results

def create_pdf_report(results, filename="fake_news_report.pdf"):
    """Create PDF report of results."""
    if not PDF_AVAILABLE:
        st.error("PDF generation not available. Install reportlab: pip install reportlab")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("🔍 Fake News Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Total Analyses: {len(results)}", styles['Normal']))
        
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        real_count = len(results) - fake_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        story.append(Paragraph(f"FAKE Predictions: {fake_count} ({fake_count/len(results)*100:.1f}%)", styles['Normal']))
        story.append(Paragraph(f"REAL Predictions: {real_count} ({real_count/len(results)*100:.1f}%)", styles['Normal']))
        story.append(Paragraph(f"Average Confidence: {avg_confidence:.2f}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed results
        story.append(Paragraph("Detailed Results:", styles['Heading2']))
        
        for i, result in enumerate(results):
            story.append(Paragraph(f"Analysis #{i+1}:", styles['Heading3']))
            story.append(Paragraph(f"Prediction: {result['prediction']}", styles['Normal']))
            story.append(Paragraph(f"Confidence: {result['confidence']:.2f}", styles['Normal']))
            story.append(Paragraph(f"Text: {result['original_text'][:200]}...", styles['Normal']))
            
            if result['top_features']:
                story.append(Paragraph("Top Influential Words:", styles['Heading4']))
                for feature, importance, _ in result['top_features'][:5]:
                    story.append(Paragraph(f"• {feature} (importance: {importance:.3f})", styles['Normal']))
            
            story.append(Spacer(1, 15))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None

def create_csv_report(results):
    """Create CSV report of results."""
    try:
        data = []
        for i, result in enumerate(results):
            row = {
                'analysis_id': i + 1,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['probabilities']['fake'],
                'real_probability': result['probabilities']['real'],
                'text_length': result['text_length'],
                'word_count': result['word_count'],
                'text_preview': result['original_text'][:100] + "..." if len(result['original_text']) > 100 else result['original_text']
            }
            
            # Add top features
            if result['top_features']:
                for j, (feature, importance, _) in enumerate(result['top_features'][:5]):
                    row[f'top_feature_{j+1}'] = feature
                    row[f'top_importance_{j+1}'] = importance
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    except Exception as e:
        st.error(f"CSV generation error: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced Fake News Detector</h1>
        <p>Advanced AI-powered fake news detection with batch processing and detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize NLTK
    if not download_nltk_data():
        st.error("Failed to download NLTK data. Some features may not work properly.")
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("❌ Models not loaded. Please check if model files exist in the 'models' directory.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Analysis Options")
    
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode:",
        ["Single Text", "Batch Upload", "Real-time Feed"]
    )
    
    if analysis_mode == "Single Text":
        single_text_analysis(model, vectorizer)
    elif analysis_mode == "Batch Upload":
        batch_analysis(model, vectorizer)
    elif analysis_mode == "Real-time Feed":
        realtime_feed_analysis(model, vectorizer)

def single_text_analysis(model, vectorizer):
    """Single text analysis interface."""
    st.header("📝 Single Text Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        text_input = st.text_area(
            "Enter news article or text to analyze:",
            height=200,
            placeholder="Paste your news article here..."
        )
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    result = predict_single_text(text_input, model, vectorizer)
                
                if result:
                    display_single_result(result)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.header("ℹ️ How It Works")
        st.markdown("""
        **Enhanced Features:**
        - 🎯 Real-time analysis
        - 📊 Confidence scoring
        - 🔍 Top influential words
        - 📈 Detailed metrics
        - 📄 Report generation
        
        **Model Info:**
        - Trained on 12,000+ articles
        - Multiple ML algorithms
        - Advanced text preprocessing
        - High accuracy detection
        """)

def batch_analysis(model, vectorizer):
    """Batch analysis interface."""
    st.header("📊 Batch Analysis")
    
    st.markdown("""
    Upload a CSV file with news articles for batch analysis. The CSV should have a column named 'text' or 'article'.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should contain a 'text' or 'article' column with news articles"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows from CSV")
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Find text column
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'article' in col.lower() or 'content' in col.lower()]
            
            if not text_columns:
                st.error("❌ No text column found. Please ensure your CSV has a column named 'text', 'article', or 'content'.")
                return
            
            text_column = st.selectbox("Select text column:", text_columns)
            
            # Process batch
            if st.button("🚀 Process Batch", type="primary"):
                texts = df[text_column].dropna().tolist()
                
                if not texts:
                    st.error("❌ No valid texts found in the selected column.")
                    return
                
                st.info(f"Processing {len(texts)} articles...")
                progress_bar = st.progress(0)
                
                with st.spinner("Processing articles..."):
                    results = process_batch_texts(texts, model, vectorizer, progress_bar)
                
                if results:
                    display_batch_results(results, df)
                else:
                    st.error("❌ No results generated.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def realtime_feed_analysis(model, vectorizer):
    """Real-time feed analysis interface."""
    st.header("📡 Real-time Feed Analysis")
    
    st.markdown("""
    Analyze news articles from RSS feeds in real-time.
    """)
    
    feed_urls = {
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "Reuters": "http://feeds.reuters.com/reuters/topNews",
        "The Guardian": "https://www.theguardian.com/world/rss"
    }
    
    selected_feed = st.selectbox("Select news feed:", list(feed_urls.keys()))
    num_articles = st.slider("Number of articles to analyze:", 1, 20, 5)
    
    if st.button("📡 Analyze Feed", type="primary"):
        try:
            import feedparser
            
            with st.spinner(f"Fetching articles from {selected_feed}..."):
                feed = feedparser.parse(feed_urls[selected_feed])
                
                if not feed.entries:
                    st.error(f"❌ No articles found in {selected_feed} feed.")
                    return
                
                articles = []
                for entry in feed.entries[:num_articles]:
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    articles.append({
                        'title': entry.get('title', 'No title'),
                        'text': text,
                        'link': entry.get('link', ''),
                        'published': entry.get('published', '')
                    })
                
                st.success(f"✅ Fetched {len(articles)} articles")
                
                # Process articles
                texts = [article['text'] for article in articles]
                progress_bar = st.progress(0)
                
                results = process_batch_texts(texts, model, vectorizer, progress_bar)
                
                # Display feed results
                display_feed_results(results, articles)
        
        except ImportError:
            st.error("❌ feedparser not installed. Run: pip install feedparser")
        except Exception as e:
            st.error(f"❌ Error fetching feed: {e}")

def display_single_result(result):
    """Display results for single text analysis."""
    st.header("📊 Analysis Results")
    
    # Prediction result
    if result['prediction'] == 'FAKE':
        st.markdown(f"""
        <div class="fake-alert">
            <h2>🚨 FAKE NEWS DETECTED</h2>
            <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="real-alert">
            <h2>✅ LIKELY REAL NEWS</h2>
            <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", result['prediction'])
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col3:
        st.metric("Word Count", result['word_count'])
    with col4:
        st.metric("Text Length", result['text_length'])
    
    # Probability breakdown
    st.subheader("🎯 Probability Breakdown")
    prob_df = pd.DataFrame({
        'Label': ['FAKE', 'REAL'],
        'Probability': [result['probabilities']['fake'], result['probabilities']['real']]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff6b6b', '#51cf66']
    bars = ax.bar(prob_df['Label'], prob_df['Probability'], color=colors)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, prob_df['Probability']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Top influential features
    if result['top_features']:
        st.subheader("🔍 Top 10 Most Influential Words")
        
        feature_df = pd.DataFrame(
            result['top_features'],
            columns=['Word', 'Importance', 'TF-IDF Score']
        ).head(10)
        
        st.dataframe(
            feature_df.style.format({
                'Importance': '{:.4f}',
                'TF-IDF Score': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Feature importance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_df))
        
        bars = ax.barh(y_pos, feature_df['Importance'], color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_df['Word'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Most Influential Words')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, feature_df['Importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Text processing details
    with st.expander("🔧 Text Processing Details"):
        st.subheader("Original Text:")
        st.text_area("", value=result['original_text'], height=100, disabled=True)
        
        st.subheader("Processed Text:")
        st.text_area("", value=result['processed_text'], height=100, disabled=True)
        
        st.metric("Processing Reduction", f"{(1 - result['processed_length']/result['text_length']):.1%}")

def display_batch_results(results, original_df):
    """Display results for batch analysis."""
    st.header("📊 Batch Analysis Results")
    
    # Summary metrics
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = len(results) - fake_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", len(results))
    with col2:
        st.metric("FAKE Detected", fake_count, f"{fake_count/len(results)*100:.1f}%")
    with col3:
        st.metric("REAL Detected", real_count, f"{real_count/len(results)*100:.1f}%")
    with col4:
        st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    results_data = []
    for i, result in enumerate(results):
        results_data.append({
            'ID': i + 1,
            'Prediction': result['prediction'],
            'Confidence': f"{result['confidence']:.1%}",
            'Top Words': ', '.join([f[0] for f in result['top_features'][:3]]) if result['top_features'] else '',
            'Text Preview': result['original_text'][:100] + "..." if len(result['original_text']) > 100 else result['original_text']
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['FAKE', 'REAL']
        sizes = [fake_count, real_count]
        colors = ['#ff6b6b', '#51cf66']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Prediction Distribution')
        st.pyplot(fig)
    
    with col2:
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        ax.axvline(avg_confidence, color='red', linestyle='--', label=f'Average: {avg_confidence:.2f}')
        ax.legend()
        st.pyplot(fig)
    
    # Download reports
    st.subheader("📥 Download Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = create_csv_report(results)
        if csv_data:
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # PDF download
        if PDF_AVAILABLE:
            pdf_data = create_pdf_report(results)
            if pdf_data:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("📄 PDF reports require reportlab: pip install reportlab")

def display_feed_results(results, articles):
    """Display results for real-time feed analysis."""
    st.header("📡 Feed Analysis Results")
    
    # Summary
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = len(results) - fake_count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Articles Analyzed", len(results))
    with col2:
        st.metric("FAKE Detected", fake_count, f"{fake_count/len(results)*100:.1f}%")
    with col3:
        st.metric("REAL Detected", real_count, f"{real_count/len(results)*100:.1f}%")
    
    # Individual results
    for i, (result, article) in enumerate(zip(results, articles)):
        with st.expander(f"📰 Article {i+1}: {article['title'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Published:** {article.get('published', 'Unknown')}")
                if article.get('link'):
                    st.write(f"**Source:** [Read Full Article]({article['link']})")
                
                # Prediction
                if result['prediction'] == 'FAKE':
                    st.error(f"🚨 FAKE (Confidence: {result['confidence']:.1%})")
                else:
                    st.success(f"✅ REAL (Confidence: {result['confidence']:.1%})")
            
            with col2:
                # Top features
                if result['top_features']:
                    st.write("**Top Words:**")
                    for word, importance, _ in result['top_features'][:5]:
                        st.write(f"• {word} ({importance:.3f})")

if __name__ == "__main__":
    main()
