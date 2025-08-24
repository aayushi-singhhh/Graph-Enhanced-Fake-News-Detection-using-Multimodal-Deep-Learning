import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import os
from datetime import datetime

# LIME and SHAP imports
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
    print("✅ LIME available")
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

# Text preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class FakeNewsExplainer:
    """
    Explainability class for fake news detection models using LIME and SHAP.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize with model and vectorizer paths."""
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Load model and vectorizer
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
        else:
            self.load_latest_model()
    
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
            model_files = glob.glob('../models/enhanced/best_fake_news_model_enhanced_*.joblib')
            vectorizer_files = glob.glob('../models/enhanced/tfidf_vectorizer_enhanced_*.joblib')
            
            if not model_files or not vectorizer_files:
                # Fallback to regular models
                model_files = glob.glob('../models/best_fake_news_model.joblib')
                vectorizer_files = glob.glob('../models/tfidf_vectorizer.joblib')
            
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
            self.model_path = model_path
            self.vectorizer_path = vectorizer_path
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
    
    def predict_proba(self, texts):
        """Prediction function for LIME."""
        if self.model is None or self.vectorizer is None:
            raise Exception("Model or vectorizer not loaded")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform and predict
        features = self.vectorizer.transform(processed_texts)
        probabilities = self.model.predict_proba(features)
        
        return probabilities
    
    def explain_with_lime(self, text, num_features=10, num_samples=1000):
        """Generate LIME explanation for a text."""
        if not LIME_AVAILABLE:
            print("❌ LIME not available")
            return None
        
        if self.model is None or self.vectorizer is None:
            print("❌ Model not loaded")
            return None
        
        try:
            # Create LIME explainer
            explainer = LimeTextExplainer(
                class_names=['FAKE', 'REAL'],
                feature_selection='auto',
                split_expr=' ',
                bow=False
            )
            
            # Generate explanation
            explanation = explainer.explain_instance(
                text,
                self.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            
            return explanation
        
        except Exception as e:
            print(f"❌ LIME explanation failed: {str(e)}")
            return None
    
    def explain_with_shap(self, text, background_samples=100):
        """Generate SHAP explanation for a text."""
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available")
            return None
        
        if self.model is None or self.vectorizer is None:
            print("❌ Model not loaded")
            return None
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            text_features = self.vectorizer.transform([processed_text])
            
            # Create a small background dataset for SHAP
            # For demonstration, we'll use a simple approach
            if hasattr(self.model, 'coef_'):
                # Linear model - use LinearExplainer
                # Create background data (zeros for sparse matrix)
                background = np.zeros((background_samples, text_features.shape[1]))
                explainer = shap.LinearExplainer(self.model, background)
                shap_values = explainer.shap_values(text_features.toarray())
            elif hasattr(self.model, 'feature_importances_'):
                # Tree model - use TreeExplainer
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(text_features)
            else:
                print("❌ Model type not supported for SHAP")
                return None
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'features': text_features,
                'processed_text': processed_text
            }
        
        except Exception as e:
            print(f"❌ SHAP explanation failed: {str(e)}")
            return None
    
    def visualize_lime_explanation(self, explanation, save_path=None):
        """Visualize LIME explanation."""
        if explanation is None:
            return
        
        try:
            # Get explanation data
            exp_data = explanation.as_list()
            
            # Separate positive and negative contributions
            words = [item[0] for item in exp_data]
            scores = [item[1] for item in exp_data]
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color by contribution
            colors = ['red' if score < 0 else 'green' for score in scores]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Feature Importance (Red=Fake, Green=Real)')
            ax.set_title('LIME Explanation: Word Contributions to Prediction', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left' if width >= 0 else 'right', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ LIME explanation saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error visualizing LIME explanation: {str(e)}")
    
    def get_top_features(self, explanation, n=10):
        """Get top contributing features from LIME explanation."""
        if explanation is None:
            return []
        
        try:
            exp_data = explanation.as_list()
            # Sort by absolute importance
            sorted_features = sorted(exp_data, key=lambda x: abs(x[1]), reverse=True)
            return sorted_features[:n]
        except:
            return []
    
    def comprehensive_analysis(self, text, save_plots=True):
        """Perform comprehensive explainability analysis."""
        print("🔍 Comprehensive Explainability Analysis")
        print("=" * 60)
        
        # Make prediction first
        try:
            prediction_proba = self.predict_proba(text)[0]
            prediction = np.argmax(prediction_proba)
            confidence = max(prediction_proba)
            
            print(f"📊 Prediction Results:")
            print(f"   Label: {'FAKE' if prediction == 0 else 'REAL'}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Probabilities: Fake={prediction_proba[0]:.3f}, Real={prediction_proba[1]:.3f}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return
        
        # LIME Analysis
        print(f"\n🟢 LIME Analysis:")
        print("-" * 30)
        
        if LIME_AVAILABLE:
            lime_explanation = self.explain_with_lime(text)
            
            if lime_explanation:
                print("✅ LIME explanation generated")
                
                # Get top features
                top_features = self.get_top_features(lime_explanation, 10)
                print(f"Top 10 contributing words:")
                for i, (word, score) in enumerate(top_features, 1):
                    direction = "→FAKE" if score < 0 else "→REAL"
                    print(f"   {i:2d}. {word:<15} {score:+.3f} {direction}")
                
                # Visualize
                if save_plots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"explanations/lime_explanation_{timestamp}.png"
                    os.makedirs("explanations", exist_ok=True)
                    self.visualize_lime_explanation(lime_explanation, save_path)
                else:
                    self.visualize_lime_explanation(lime_explanation)
        else:
            print("❌ LIME not available")
        
        # SHAP Analysis
        print(f"\n🔵 SHAP Analysis:")
        print("-" * 30)
        
        if SHAP_AVAILABLE:
            shap_result = self.explain_with_shap(text)
            
            if shap_result:
                print("✅ SHAP explanation generated")
                print("   SHAP values computed for feature importance")
                
                # For tree models, we can show feature importance
                if hasattr(self.model, 'feature_importances_'):
                    feature_names = self.vectorizer.get_feature_names_out()
                    if hasattr(shap_result['shap_values'], 'shape') and len(shap_result['shap_values'].shape) > 1:
                        # Get SHAP values for the prediction class
                        values = shap_result['shap_values'][0] if isinstance(shap_result['shap_values'], list) else shap_result['shap_values'][0]
                        
                        # Get non-zero features
                        non_zero_indices = np.nonzero(shap_result['features'].toarray()[0])[0]
                        if len(non_zero_indices) > 0:
                            feature_values = [(feature_names[i], values[i]) for i in non_zero_indices]
                            feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            print("Top SHAP feature contributions:")
                            for i, (feature, value) in enumerate(feature_values[:10], 1):
                                direction = "→FAKE" if value < 0 else "→REAL"
                                print(f"   {i:2d}. {feature:<15} {value:+.3f} {direction}")
            else:
                print("❌ SHAP explanation failed")
        else:
            print("❌ SHAP not available")
        
        print(f"\n✅ Comprehensive analysis completed!")

def main():
    """Main function for testing explainability."""
    
    # Create explainer
    explainer = FakeNewsExplainer()
    
    if explainer.model is None:
        print("❌ Could not load model. Please ensure model files exist.")
        return
    
    # Test articles
    test_articles = [
        {
            'text': "Scientists at Stanford University published a peer-reviewed study showing significant improvements in cancer treatment using immunotherapy. The research was conducted over 5 years with 1000 patients.",
            'description': 'Scientific news article (likely real)'
        },
        {
            'text': "BREAKING: World Health Organization confirms that drinking bleach cures coronavirus! Doctors don't want you to know this simple trick! Share before they delete this information!",
            'description': 'Sensational health misinformation (likely fake)'
        }
    ]
    
    print("🧪 Testing Explainability Features")
    print("=" * 60)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n📰 Article {i}: {article['description']}")
        print(f"Text: {article['text'][:100]}...")
        
        # Perform comprehensive analysis
        explainer.comprehensive_analysis(article['text'])
        
        if i < len(test_articles):
            input("Press Enter to continue to next article...")

if __name__ == "__main__":
    main()
