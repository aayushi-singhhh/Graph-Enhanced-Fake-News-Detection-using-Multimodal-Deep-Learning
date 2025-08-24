from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# Text preprocessing imports
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ NLTK data downloaded successfully")
except:
    print("⚠️ Warning: Could not download NLTK data")

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="AI-powered API for detecting fake news articles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_info = {}

# Pydantic models
class PredictionRequest(BaseModel):
    text: str
    return_details: bool = True

class PredictionResponse(BaseModel):
    prediction: Dict[str, Any]
    text_analysis: Dict[str, Any]
    model_info: Dict[str, str]
    details: Optional[Dict[str, str]] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    model_info: Optional[Dict[str, Any]] = None

# Enhanced preprocessing function
def enhanced_preprocessing(text: str) -> str:
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
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

# Load model function
def load_model():
    """Load the latest trained model and vectorizer."""
    global model, vectorizer, model_info
    
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
        
        model = joblib.load(latest_model)
        vectorizer = joblib.load(latest_vectorizer)
        
        model_info = {
            'model_file': os.path.basename(latest_model),
            'model_type': type(model).__name__,
            'vocabulary_size': len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 'Unknown',
            'loaded_at': datetime.now().isoformat()
        }
        
        print(f"✅ Model loaded successfully: {model_info['model_file']}")
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Prediction function
def predict_fake_news(text: str, return_details: bool = True) -> Dict[str, Any]:
    """Enhanced production-ready fake news prediction function."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess text
        processed_text = enhanced_preprocessing(text)
        
        if not processed_text:
            raise HTTPException(
                status_code=422, 
                detail='Text preprocessing resulted in empty content'
            )
        
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
            'model_info': {
                'model_type': model_info.get('model_type', 'Unknown'),
                'model_file': model_info.get('model_file', 'Unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if return_details:
            result['details'] = {
                'original_text': text,
                'processed_text': processed_text
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')

# API Routes
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("🚀 Starting Fake News Detection API...")
    if not load_model():
        print("⚠️ Warning: Could not load model on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        model_info=model_info if model is not None else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Main prediction endpoint."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    
    if len(request.text) > 10000:  # Limit text length
        raise HTTPException(status_code=422, detail="Text too long (max 10,000 characters)")
    
    result = predict_fake_news(request.text, request.return_details)
    
    return PredictionResponse(**result)

@app.post("/predict/simple")
async def predict_simple(request: PredictionRequest):
    """Simplified prediction endpoint returning only label and confidence."""
    result = predict_fake_news(request.text, return_details=False)
    
    return {
        "label": result['prediction']['label'],
        "confidence": result['prediction']['confidence'],
        "probabilities": result['prediction']['probabilities']
    }

@app.get("/model/info")
async def model_info_endpoint():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_info

@app.post("/model/reload")
async def reload_model():
    """Reload the model (useful for model updates)."""
    if load_model():
        return {"message": "Model reloaded successfully", "model_info": model_info}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
