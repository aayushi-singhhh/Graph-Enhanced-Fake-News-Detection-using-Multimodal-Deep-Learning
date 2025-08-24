import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import feedparser
from urllib.parse import urljoin
import joblib
import glob

# Text preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class RealDataEvaluator:
    """
    Evaluate fake news detection model on real-time news data.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize with model paths."""
        self.model = None
        self.vectorizer = None
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Load model
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
        else:
            self.load_latest_model()
        
        # RSS feeds for real news sources
        self.news_feeds = {
            'reuters': 'http://feeds.reuters.com/reuters/topNews',
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'cnn': 'http://rss.cnn.com/rss/edition.rss',
            'npr': 'https://feeds.npr.org/1001/rss.xml',
            'associated_press': 'https://rsshub.app/ap/topics/apf-topnews',
            'guardian': 'https://www.theguardian.com/uk/rss'
        }
        
        # Known reliable vs unreliable domains for validation
        self.reliable_domains = {
            'reuters.com', 'bbc.com', 'cnn.com', 'npr.org', 
            'apnews.com', 'theguardian.com', 'nytimes.com',
            'washingtonpost.com', 'wsj.com', 'abcnews.go.com'
        }
        
        self.unreliable_patterns = {
            'conspiracy', 'truthteller', 'realpatriotnews', 'infowars',
            'naturalnews', 'beforeitsnews', 'globalresearch'
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            print("⚠️ Warning: Could not download NLTK data")
    
    def load_latest_model(self):
        """Load the latest model and vectorizer."""
        try:
            # Find the latest enhanced model files
            model_files = glob.glob('models/enhanced/best_fake_news_model_enhanced_*.joblib')
            vectorizer_files = glob.glob('models/enhanced/tfidf_vectorizer_enhanced_*.joblib')
            
            if not model_files or not vectorizer_files:
                # Fallback to regular models
                model_files = glob.glob('models/best_fake_news_model.joblib')
                vectorizer_files = glob.glob('models/tfidf_vectorizer.joblib')
            
            if not model_files or not vectorizer_files:
                raise Exception("No model files found")
            
            # Load the latest files
            latest_model = max(model_files, key=os.path.getctime)
            latest_vectorizer = max(vectorizer_files, key=os.path.getctime)
            
            self.load_model(latest_model, latest_vectorizer)
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def load_model(self, model_path, vectorizer_path):
        """Load specific model and vectorizer."""
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"✅ Model loaded: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
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
    
    def predict_fake_news(self, text):
        """Predict if news is fake or real."""
        if self.model is None or self.vectorizer is None:
            return None
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return None
            
            # Transform and predict
            text_features = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_features)[0]
            prediction_proba = self.model.predict_proba(text_features)[0]
            
            return {
                'label': 'FAKE' if prediction == 0 else 'REAL',
                'prediction': int(prediction),
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    'fake': float(prediction_proba[0]),
                    'real': float(prediction_proba[1])
                }
            }
        
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None
    
    def fetch_rss_articles(self, feed_url, max_articles=10):
        """Fetch articles from RSS feed."""
        try:
            print(f"🔄 Fetching articles from: {feed_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown')
                }
                
                # Combine title and description
                article['content'] = f"{article['title']} {article['description']}"
                articles.append(article)
            
            print(f"✅ Fetched {len(articles)} articles")
            return articles
        
        except Exception as e:
            print(f"❌ Error fetching RSS feed {feed_url}: {str(e)}")
            return []
    
    def evaluate_domain_reliability(self, url):
        """Evaluate if a domain is typically reliable."""
        try:
            domain = url.split('/')[2].lower()
            
            # Check if domain is in reliable list
            if any(reliable in domain for reliable in self.reliable_domains):
                return 'reliable'
            
            # Check for unreliable patterns
            if any(pattern in domain for pattern in self.unreliable_patterns):
                return 'unreliable'
            
            return 'unknown'
        
        except:
            return 'unknown'
    
    def collect_real_time_news(self, max_articles_per_feed=5):
        """Collect real-time news from multiple sources."""
        print("📰 Collecting Real-Time News Articles")
        print("=" * 50)
        
        all_articles = []
        
        for source_name, feed_url in self.news_feeds.items():
            print(f"\n📡 {source_name.upper()}")
            print("-" * 30)
            
            articles = self.fetch_rss_articles(feed_url, max_articles_per_feed)
            
            for article in articles:
                article['source_name'] = source_name
                article['domain_reliability'] = self.evaluate_domain_reliability(article['link'])
                all_articles.append(article)
            
            # Be respectful to servers
            time.sleep(1)
        
        print(f"\n✅ Total articles collected: {len(all_articles)}")
        return all_articles
    
    def evaluate_on_real_data(self, articles):
        """Evaluate model performance on real news data."""
        print("\n🔍 Evaluating Model on Real-Time Data")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Model not loaded")
            return
        
        results = []
        
        for i, article in enumerate(articles, 1):
            print(f"\n📰 Article {i}/{len(articles)}")
            print(f"Source: {article['source_name']}")
            print(f"Title: {article['title'][:100]}...")
            print(f"Domain Reliability: {article['domain_reliability']}")
            
            # Predict
            prediction = self.predict_fake_news(article['content'])
            
            if prediction:
                article['ai_prediction'] = prediction
                
                print(f"AI Prediction: {prediction['label']} (Confidence: {prediction['confidence']:.3f})")
                
                # Compare with domain reliability
                if article['domain_reliability'] == 'reliable' and prediction['label'] == 'FAKE':
                    print("⚠️  Warning: Reliable source predicted as FAKE")
                elif article['domain_reliability'] == 'unreliable' and prediction['label'] == 'REAL':
                    print("⚠️  Warning: Unreliable source predicted as REAL")
                elif article['domain_reliability'] == 'reliable' and prediction['label'] == 'REAL':
                    print("✅ Consistent: Reliable source predicted as REAL")
                
                results.append(article)
            else:
                print("❌ Prediction failed")
        
        return results
    
    def analyze_results(self, results):
        """Analyze and summarize evaluation results."""
        print("\n📊 Analysis Summary")
        print("=" * 50)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Extract prediction information
        df['ai_label'] = df['ai_prediction'].apply(lambda x: x['label'] if x else None)
        df['ai_confidence'] = df['ai_prediction'].apply(lambda x: x['confidence'] if x else None)
        df['fake_prob'] = df['ai_prediction'].apply(lambda x: x['probabilities']['fake'] if x else None)
        df['real_prob'] = df['ai_prediction'].apply(lambda x: x['probabilities']['real'] if x else None)
        
        # Summary statistics
        print(f"📈 Overall Statistics:")
        print(f"   - Total articles analyzed: {len(df)}")
        print(f"   - Predicted as REAL: {(df['ai_label'] == 'REAL').sum()}")
        print(f"   - Predicted as FAKE: {(df['ai_label'] == 'FAKE').sum()}")
        print(f"   - Average confidence: {df['ai_confidence'].mean():.3f}")
        
        # By source analysis
        print(f"\n📊 By Source Analysis:")
        source_analysis = df.groupby('source_name').agg({
            'ai_label': lambda x: (x == 'REAL').sum(),
            'ai_confidence': 'mean'
        }).round(3)
        source_analysis.columns = ['Predicted_REAL', 'Avg_Confidence']
        print(source_analysis.to_string())
        
        # Domain reliability analysis
        print(f"\n🔍 Domain Reliability Analysis:")
        if 'domain_reliability' in df.columns:
            reliability_analysis = df.groupby(['domain_reliability', 'ai_label']).size().unstack(fill_value=0)
            print(reliability_analysis.to_string())
            
            # Calculate consistency score
            reliable_real = df[(df['domain_reliability'] == 'reliable') & (df['ai_label'] == 'REAL')].shape[0]
            total_reliable = df[df['domain_reliability'] == 'reliable'].shape[0]
            
            if total_reliable > 0:
                consistency_score = reliable_real / total_reliable
                print(f"\n📊 Consistency Score: {consistency_score:.3f}")
                print(f"   (Reliable sources predicted as REAL / Total reliable sources)")
            
        # High-confidence predictions
        high_conf_threshold = 0.8
        high_conf_articles = df[df['ai_confidence'] >= high_conf_threshold]
        print(f"\n🎯 High-Confidence Predictions (≥{high_conf_threshold}):")
        print(f"   - Count: {len(high_conf_articles)}")
        if len(high_conf_articles) > 0:
            print(f"   - Real: {(high_conf_articles['ai_label'] == 'REAL').sum()}")
            print(f"   - Fake: {(high_conf_articles['ai_label'] == 'FAKE').sum()}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/real_data_evaluation_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to: {results_path}")
        
        return df
    
    def run_evaluation(self, max_articles_per_feed=5):
        """Run complete real-data evaluation."""
        print("🚀 Real-Data Evaluation Pipeline")
        print("=" * 60)
        
        if self.model is None:
            print("❌ Model not loaded. Cannot proceed.")
            return
        
        # Collect articles
        articles = self.collect_real_time_news(max_articles_per_feed)
        
        if not articles:
            print("❌ No articles collected")
            return
        
        # Evaluate
        results = self.evaluate_on_real_data(articles)
        
        if not results:
            print("❌ No evaluation results")
            return
        
        # Analyze
        analysis_df = self.analyze_results(results)
        
        print(f"\n✅ Real-data evaluation completed!")
        return analysis_df

def main():
    """Main function for real data evaluation."""
    
    print("📊 Real-Time News Evaluation")
    print("=" * 50)
    
    # Create evaluator
    evaluator = RealDataEvaluator()
    
    if evaluator.model is None:
        print("❌ Could not load model. Please ensure model files exist.")
        return
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(max_articles_per_feed=3)
        
        if results is not None:
            print(f"\n🎉 Evaluation completed successfully!")
            print(f"📊 Analyzed {len(results)} articles from real news sources")
        else:
            print("❌ Evaluation failed")
    
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main()
