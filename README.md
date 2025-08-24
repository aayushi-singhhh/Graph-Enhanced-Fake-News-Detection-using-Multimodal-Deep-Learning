# 🔍 Graph-Enhanced Fake News Detection using Multimodal Deep Learning

A comprehensive, production-ready fake news detection system using advanced machine learning techniques, featuring deep learning models, explainability tools, and real-time monitoring.

## 🎯 Project Overview

This project implements a complete fake news detection pipeline with:
- **Multiple ML Models**: Traditional ML + Deep Learning + Ensemble methods
- **Explainability**: LIME and SHAP integration for model interpretability
- **Production Ready**: REST API, Web Interface, Docker deployment
- **Real-time Monitoring**: Performance tracking and drift detection
- **Live Evaluation**: RSS feed integration for real-world testing

## 📊 Dataset & Performance

- **Dataset Size**: 12,273 news articles with binary labels (fake/real)
- **Class Distribution**: 9.2% fake vs 90.8% real (handled with SMOTE + class balancing)
- **Best Model**: Random Forest (Balanced) with F1-Score: 0.XXX
- **Features**: TF-IDF vectorization (5,000 features, unigrams + bigrams)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning.git
cd Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Run the complete training pipeline
jupyter notebook notebooks/fake_news.ipynb

# Or train individual components
python deep_learning.py  # Deep learning models
```

### 3. Launch Applications

```bash
# Web Interface (Streamlit)
streamlit run app.py

# REST API (FastAPI)
uvicorn api.main:app --reload

# Docker Deployment
docker-compose up -d
```

## 🏗️ Architecture

```
📁 Project Structure
├── 📊 notebooks/
│   └── fake_news.ipynb          # Main training pipeline
├── 🤖 models/
│   ├── enhanced/                # Trained models & metadata
│   └── *.joblib                 # Model files
├── 🌐 api/
│   ├── main.py                  # FastAPI backend
│   └── test_api.py              # API testing
├── 📱 Frontend
│   └── app.py                   # Streamlit web interface
├── 🔍 Analysis & Monitoring
│   ├── explainability.py       # LIME/SHAP explanations
│   ├── deep_learning.py        # Advanced ML models
│   ├── real_data_evaluation.py # Live news testing
│   └── production_monitoring.py # Model monitoring
├── 🐳 Deployment
│   ├── Dockerfile               # Container configuration
│   ├── docker-compose.yml      # Multi-service deployment
│   └── requirements.txt        # Dependencies
└── 📋 Documentation
    └── README.md                # This file
```

## 🤖 Model Zoo

### Traditional ML Models
- **Logistic Regression** (Baseline, Balanced, SMOTE)
- **Random Forest** (Balanced, SMOTE) ⭐ Best Performance
- **XGBoost** (Balanced)
- **Ensemble** (Voting Classifier)

### Deep Learning Models
- **LSTM** - Sequential text modeling
- **BiLSTM** - Bidirectional context understanding
- **CNN** - Convolutional text classification
- **BERT/DistilBERT** - Transformer-based (optional)

### Explainability Tools
- **LIME** - Local feature importance
- **SHAP** - Global and local explanations
- **Feature Importance** - Tree-based model insights

## 🌐 API Endpoints

### FastAPI Backend (`http://localhost:8000`)

```bash
# Health Check
GET /health

# Predict (Full Response)
POST /predict
{
  "text": "Your news article text here",
  "return_details": true
}

# Predict (Simple Response)
POST /predict/simple
{
  "text": "Your news article text here"
}

# Model Information
GET /model/info

# Reload Model
POST /model/reload
```

### Response Format
```json
{
  "prediction": {
    "label": "REAL",
    "confidence": 0.85,
    "probabilities": {
      "fake": 0.15,
      "real": 0.85
    }
  },
  "text_analysis": {
    "original_length": 150,
    "processed_length": 120,
    "word_count": 25
  },
  "model_info": {
    "model_type": "RandomForestClassifier",
    "model_file": "best_fake_news_model_enhanced_20250825_015422.joblib"
  },
  "timestamp": "2025-08-25T01:54:22"
}
```

## 🔍 Explainability Features

### LIME Explanations
```python
from explainability import FakeNewsExplainer

explainer = FakeNewsExplainer()
explainer.comprehensive_analysis("Your news article text")
```

### SHAP Values
```python
# Integrated in the explainability module
# Provides feature importance for individual predictions
```

## 📊 Monitoring & Production

### Real-time Monitoring
```python
from production_monitoring import ModelMonitor

monitor = ModelMonitor()
# Tracks performance, drift, alerts
dashboard = monitor.create_monitoring_dashboard()
```

### Live News Evaluation
```python
from real_data_evaluation import RealDataEvaluator

evaluator = RealDataEvaluator()
results = evaluator.run_evaluation()
```

### Key Monitoring Metrics
- **Performance Drift**: Accuracy, precision, recall, F1-score changes
- **Data Drift**: Input distribution changes over time
- **Response Time**: API latency monitoring
- **Confidence Levels**: Model uncertainty tracking
- **Prediction Patterns**: Fake news ratio trends

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t fake-news-detector .

# Run container
docker run -p 8000:8000 -v ./models:/app/models fake-news-detector
```

### Multi-Service (Recommended)
```bash
# Launch all services
docker-compose up -d

# Services available:
# - API: http://localhost:8000
# - Web App: http://localhost:8501
```

## 🧪 Testing & Validation

### API Testing
```bash
python api/test_api.py
```

### Model Evaluation
```bash
# Test on real RSS feeds
python real_data_evaluation.py

# Performance monitoring
python production_monitoring.py
```

### Sample Predictions
```python
# Likely REAL news
"Scientists at Stanford University published a peer-reviewed study..."

# Likely FAKE news  
"BREAKING: World Health Organization confirms drinking bleach cures coronavirus!"
```

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Baseline) | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest (Balanced) ⭐ | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| XGBoost (Balanced) | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| LSTM | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| BiLSTM | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Ensemble | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/enhanced/best_fake_news_model_enhanced_*.joblib

# Monitoring
LOG_LEVEL=INFO
MONITORING_INTERVAL=3600
ALERT_THRESHOLD=0.05
```

### Model Parameters
```python
# TF-IDF Configuration
MAX_FEATURES = 5000
MIN_DF = 5
MAX_DF = 0.8
NGRAM_RANGE = (1, 2)

# Model Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

## 🚨 Production Considerations

### Security
- [ ] Input validation and sanitization
- [ ] Rate limiting on API endpoints
- [ ] Authentication for admin endpoints
- [ ] HTTPS in production deployment

### Scalability
- [ ] Model caching for faster inference
- [ ] Database integration for prediction logging
- [ ] Load balancer for multiple API instances
- [ ] GPU acceleration for deep learning models

### Monitoring
- [ ] Automated retraining pipeline
- [ ] A/B testing for model updates
- [ ] Performance alerting system
- [ ] Data quality checks

## 🔮 Future Enhancements

### Technical Improvements
- [ ] **Multi-modal Analysis**: Image and video content detection
- [ ] **Graph Neural Networks**: Social network analysis integration
- [ ] **Real-time Training**: Online learning capabilities
- [ ] **Federated Learning**: Privacy-preserving collaborative training

### Feature Additions
- [ ] **Browser Extension**: Real-time fact-checking while browsing
- [ ] **Social Media Integration**: Twitter/Facebook API integration
- [ ] **Multilingual Support**: Multiple language detection
- [ ] **Source Credibility**: Publisher reputation scoring

### Infrastructure
- [ ] **Kubernetes Deployment**: Container orchestration
- [ ] **Cloud Integration**: AWS/GCP/Azure deployment
- [ ] **CDN Integration**: Global content delivery
- [ ] **Microservices Architecture**: Scalable service decomposition

## 📚 Research & References

### Key Papers
- "Attention Is All You Need" (Transformer Architecture)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Fake News Detection using Machine Learning"

### Datasets
- **FakeNewsNet**: Multi-modal fake news detection
- **LIAR**: Political fact-checking dataset
- **COVID-19 Infodemic**: Pandemic misinformation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black . && flake8 .
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning framework
- **TensorFlow/Keras**: Deep learning models
- **Streamlit**: Web interface framework
- **FastAPI**: High-performance API framework
- **LIME/SHAP**: Explainability libraries
- **NLTK**: Natural language processing

## 📞 Contact & Support

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **Issues**: [GitHub Issues](https://github.com/your-username/Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning/issues)

---

**⚠️ Disclaimer**: This tool provides AI-based suggestions for fake news detection. Always verify information with multiple reliable sources and use critical thinking when evaluating news content. The tool should be used as part of a broader fact-checking approach, not as the sole determinant of news authenticity.
