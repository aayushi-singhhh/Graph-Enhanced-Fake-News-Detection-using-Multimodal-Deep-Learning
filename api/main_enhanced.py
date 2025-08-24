"""
Enhanced FastAPI Backend for Fake News Detection
with improved error handling, top features, and comprehensive responses
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
import time
from collections import Counter
import traceback

# Text preprocessing imports
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Language detection
try:
    from langdetect import detect, LangDetectError
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    logger.info("✅ NLTK data downloaded successfully")
except:
    logger.warning("⚠️ Warning: Could not download NLTK data")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Fake News Detection API",
    description="Advanced AI-powered API for detecting fake news with detailed analysis",
    version="2.0.0",
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
    include_top_features: bool = True
    max_features: int = 10
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 50000:  # Limit text length
            raise ValueError('Text too long (max 50,000 characters)')
        return v.strip()
    
    @validator('max_features')
    def validate_max_features(cls, v):
        if v < 1 or v > 50:
            raise ValueError('max_features must be between 1 and 50')
        return v

class FeatureInfo(BaseModel):
    word: str
    importance: float
    frequency: float

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    top_features: Optional[List[FeatureInfo]] = None
    text_analysis: Dict[str, Any]
    model_info: Dict[str, str]
    processing_time: float
    timestamp: str
    warnings: List[str] = []

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    return_details: bool = True
    include_top_features: bool = False
    max_features: int = 5
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('texts list cannot be empty')
        if len(v) > 100:  # Limit batch size
            raise ValueError('Maximum 100 texts per batch')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return v

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, Any]
    total_processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    model_info: Optional[Dict[str, Any]] = None
    system_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            message=f"Request validation failed: {str(exc)}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# Enhanced preprocessing function
def enhanced_preprocessing(text: str) -> tuple[str, List[str]]:
    """Enhanced text preprocessing with warning collection."""
    warnings = []
    
    if pd.isna(text) or text == '':
        warnings.append("Empty text provided")
        return "", warnings
    
    original_length = len(text)
    
    try:
        # Language detection
        if LANG_DETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                if detected_lang != 'en':
                    warnings.append(f"Non-English text detected: {detected_lang}")
            except LangDetectError:
                warnings.append("Could not detect language")
        
        # Initialize preprocessing tools
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Preprocessing steps
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove social media handles and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove numbers (but keep some context)
        text = re.sub(r'\d+', ' number ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if token.lower() not in stop_words and len(token) > 2 and token.isalpha():
                filtered_tokens.append(lemmatizer.lemmatize(token.lower()))
        
        processed_text = ' '.join(filtered_tokens)
        
        # Check processing effectiveness
        reduction_ratio = 1 - len(processed_text) / original_length if original_length > 0 else 0
        if reduction_ratio > 0.8:
            warnings.append("Significant text reduction during preprocessing")
        
        if len(filtered_tokens) < 5:
            warnings.append("Very few words remaining after preprocessing")
        
        return processed_text, warnings
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text preprocessing failed: {str(e)}")

def get_top_features(model, vectorizer, text: str, processed_text: str, max_features: int = 10) -> List[FeatureInfo]:
    """Extract top influential features for the prediction."""
    try:
        # Transform text
        text_vector = vectorizer.transform([processed_text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features
        nonzero_indices = text_vector.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return []
        
        features = []
        
        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            for idx in nonzero_indices:
                feature_name = feature_names[idx]
                importance = model.feature_importances_[idx]
                frequency = text_vector[0, idx]
                
                features.append(FeatureInfo(
                    word=feature_name,
                    importance=float(importance),
                    frequency=float(frequency)
                ))
        
        # For linear models
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            for idx in nonzero_indices:
                feature_name = feature_names[idx]
                coefficient = coef[idx]
                frequency = text_vector[0, idx]
                importance = abs(coefficient * frequency)
                
                features.append(FeatureInfo(
                    word=feature_name,
                    importance=float(importance),
                    frequency=float(frequency)
                ))
        
        # Sort by importance and return top features
        features.sort(key=lambda x: x.importance, reverse=True)
        return features[:max_features]
    
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return []

def predict_text(text: str, include_top_features: bool = True, max_features: int = 10) -> Dict[str, Any]:
    """Make prediction on text with comprehensive analysis."""
    start_time = time.time()
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preprocess text
    processed_text, warnings = enhanced_preprocessing(text)
    
    if not processed_text:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing")
    
    try:
        # Transform text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get top features if requested
        top_features = None
        if include_top_features:
            top_features = get_top_features(model, vectorizer, text, processed_text, max_features)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        result = {
            'text': text,
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'top_features': top_features,
            'text_analysis': {
                'original_length': len(text),
                'processed_length': len(processed_text),
                'word_count': len(text.split()),
                'processed_word_count': len(processed_text.split()),
                'preprocessing_reduction': 1 - len(processed_text) / len(text) if len(text) > 0 else 0
            },
            'model_info': model_info,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'warnings': warnings
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Load model function
def load_models():
    """Load the trained model and vectorizer."""
    global model, vectorizer, model_info
    
    try:
        # Find model files
        model_files = glob.glob('../models/**/best_fake_news_model*.joblib', recursive=True)
        if not model_files:
            model_files = glob.glob('../models/best_fake_news_model*.joblib')
        if not model_files:
            model_files = glob.glob('models/**/best_fake_news_model*.joblib', recursive=True)
        if not model_files:
            model_files = glob.glob('models/best_fake_news_model*.joblib')
        
        if model_files:
            model_path = model_files[-1]  # Get latest model
            model = joblib.load(model_path)
            logger.info(f"✅ Model loaded from: {model_path}")
        else:
            logger.error("❌ No model file found")
            return False
        
        # Find vectorizer files
        vectorizer_files = glob.glob('../models/**/tfidf_vectorizer*.joblib', recursive=True)
        if not vectorizer_files:
            vectorizer_files = glob.glob('../models/tfidf_vectorizer*.joblib')
        if not vectorizer_files:
            vectorizer_files = glob.glob('models/**/tfidf_vectorizer*.joblib', recursive=True)
        if not vectorizer_files:
            vectorizer_files = glob.glob('models/tfidf_vectorizer*.joblib')
        
        if vectorizer_files:
            vectorizer_path = vectorizer_files[-1]
            vectorizer = joblib.load(vectorizer_path)
            logger.info(f"✅ Vectorizer loaded from: {vectorizer_path}")
        else:
            logger.error("❌ No vectorizer file found")
            return False
        
        # Store model info
        model_info = {
            'model_type': type(model).__name__,
            'model_file': os.path.basename(model_path),
            'vectorizer_file': os.path.basename(vectorizer_path),
            'features_count': str(vectorizer.vocabulary_.__len__() if hasattr(vectorizer, 'vocabulary_') else 'unknown'),
            'loaded_at': datetime.now().isoformat()
        }
        
        return True
    
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("🚀 Starting Enhanced Fake News Detection API...")
    success = load_models()
    if success:
        logger.info("✅ API ready to serve requests")
    else:
        logger.error("❌ API started but models not loaded")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Enhanced Fake News Detection API",
        "version": "2.0.0",
        "status": "operational",
        "model_loaded": model is not None and vectorizer is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None and vectorizer is not None else "unhealthy",
        model_loaded=model is not None and vectorizer is not None,
        timestamp=datetime.now().isoformat(),
        model_info=model_info if model_info else None,
        system_info={
            "python_version": "3.9+",
            "nltk_available": True,
            "language_detection": LANG_DETECT_AVAILABLE,
            "uptime": "running"
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fake_news(request: PredictionRequest):
    """Enhanced single text prediction with detailed analysis."""
    try:
        result = predict_text(
            text=request.text,
            include_top_features=request.include_top_features,
            max_features=request.max_features
        )
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_fake_news(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    start_time = time.time()
    
    try:
        results = []
        
        for text in request.texts:
            result = predict_text(
                text=text,
                include_top_features=request.include_top_features,
                max_features=request.max_features
            )
            results.append(PredictionResponse(**result))
        
        # Calculate summary statistics
        fake_count = sum(1 for r in results if r.prediction == 'FAKE')
        real_count = len(results) - fake_count
        avg_confidence = np.mean([r.confidence for r in results])
        
        total_time = time.time() - start_time
        
        summary = {
            'total_texts': len(results),
            'fake_predictions': fake_count,
            'real_predictions': real_count,
            'fake_percentage': fake_count / len(results) * 100,
            'average_confidence': float(avg_confidence),
            'average_processing_time': total_time / len(results)
        }
        
        return BatchPredictionResponse(
            results=results,
            summary=summary,
            total_processing_time=total_time,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_info.copy()
    
    # Add runtime information
    if hasattr(model, 'feature_importances_'):
        info['model_category'] = 'tree_based'
        info['supports_feature_importance'] = True
    elif hasattr(model, 'coef_'):
        info['model_category'] = 'linear'
        info['supports_coefficients'] = True
    else:
        info['model_category'] = 'other'
    
    info['prediction_classes'] = ['FAKE', 'REAL']
    info['api_version'] = '2.0.0'
    
    return info

@app.get("/model/reload")
async def reload_models():
    """Reload models (admin endpoint)."""
    success = load_models()
    if success:
        return {"message": "Models reloaded successfully", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload models")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
