# 🚀 Deployment Guide

Complete deployment guide for the Fake News Detection system across different environments.

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git
- 4GB+ RAM (8GB recommended for deep learning)
- 10GB+ storage

## 🏠 Local Development

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning.git
cd Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 3. Train Models (Optional)
```bash
# Run training notebook
jupyter notebook notebooks/fake_news.ipynb

# Or use pre-trained models (if available)
```

### 4. Start Services
```bash
# Web Interface
streamlit run app.py --server.port 8501

# API Server  
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Docker Deployment

### Single Container Deployment

#### Build & Run
```bash
# Build image
docker build -t fake-news-detector .

# Run with volume mounting
docker run -d \
  --name fake-news-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fake-news-detector
```

#### Environment Variables
```bash
docker run -d \
  --name fake-news-api \
  -p 8000:8000 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/models:/app/models \
  fake-news-detector
```

### Multi-Service Deployment (Recommended)

#### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

#### Services Available
- **API Server**: http://localhost:8000
- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone <your-repo>
cd Graph-Enhanced-Fake-News-Detection-using-Multimodal-Deep-Learning
docker-compose up -d
```

#### AWS ECS (Elastic Container Service)
```yaml
# task-definition.json
{
  "family": "fake-news-detector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fake-news-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/fake-news-detector:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_HOST",
          "value": "0.0.0.0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fake-news-detector",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### AWS Lambda (Serverless)
```python
# lambda_function.py
import json
import boto3
from api.main import predict_fake_news

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        text = body['text']
        
        # Load model from S3 or container
        result = predict_fake_news(text)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/fake-news-detector

# Deploy to Cloud Run
gcloud run deploy fake-news-detector \
  --image gcr.io/PROJECT-ID/fake-news-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 8000
```

#### GKE (Kubernetes) Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fake-news-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fake-news-detector
  template:
    metadata:
      labels:
        app: fake-news-detector
    spec:
      containers:
      - name: fake-news-api
        image: gcr.io/PROJECT-ID/fake-news-detector:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: fake-news-service
spec:
  selector:
    app: fake-news-detector
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Microsoft Azure

#### Azure Container Instances
```bash
# Create resource group
az group create --name fake-news-rg --location eastus

# Deploy container
az container create \
  --resource-group fake-news-rg \
  --name fake-news-detector \
  --image your-registry/fake-news-detector:latest \
  --dns-name-label fake-news-detector \
  --ports 8000 \
  --memory 2 \
  --cpu 1
```

#### Azure App Service
```bash
# Create App Service plan
az appservice plan create \
  --name fake-news-plan \
  --resource-group fake-news-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group fake-news-rg \
  --plan fake-news-plan \
  --name fake-news-detector \
  --deployment-container-image-name your-registry/fake-news-detector:latest
```

## 🌐 Platform-as-a-Service Deployments

### Heroku
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create fake-news-detector

# Set environment variables
heroku config:set API_HOST=0.0.0.0
heroku config:set API_PORT=$PORT

# Deploy
git push heroku main
```

#### Heroku Configuration
```yaml
# heroku.yml
build:
  docker:
    web: Dockerfile
run:
  web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

### Railway
```yaml
# railway.toml
[build]
builder = "docker"

[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "always"
```

### Render
```yaml
# render.yaml
services:
  - type: web
    name: fake-news-detector
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: API_HOST
        value: 0.0.0.0
      - key: API_PORT
        value: 10000
```

## 📊 Production Configuration

### Environment Variables
```bash
# Required
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Optional
MODEL_PATH=/app/models/enhanced/
MONITORING_ENABLED=true
CACHE_TTL=3600
MAX_REQUEST_SIZE=10MB
RATE_LIMIT=100/minute

# Database (if using)
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379

# External Services
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-key
```

### Health Checks
```bash
# API Health Check
curl http://localhost:8000/health

# Model Status
curl http://localhost:8000/model/info

# Docker Health Check
docker run --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  fake-news-detector
```

### Load Balancing (Nginx)
```nginx
# nginx.conf
upstream fake_news_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://fake_news_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://fake_news_backend;
    }
}
```

## 📈 Monitoring & Logging

### Application Monitoring
```python
# monitoring_config.py
import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Sentry integration
sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
)

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
```

## 🔒 Security Considerations

### API Security
```python
# security.py
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
```

### Rate Limiting
```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_endpoint(request: Request, ...):
    ...
```

## 🧪 Testing Deployment

### Automated Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load testing
locust -f tests/load_test.py --host=http://localhost:8000
```

### Health Check Script
```bash
#!/bin/bash
# health_check.sh

echo "Testing Fake News Detection API..."

# Test health endpoint
curl -f http://localhost:8000/health || exit 1

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Test news article"}' || exit 1

echo "All health checks passed!"
```

## 🚀 Continuous Deployment

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and push Docker image
      run: |
        docker build -t fake-news-detector .
        docker tag fake-news-detector ${{ secrets.REGISTRY_URL }}/fake-news-detector:latest
        docker push ${{ secrets.REGISTRY_URL }}/fake-news-detector:latest
    
    - name: Deploy to production
      run: |
        # Deploy script here
        kubectl apply -f k8s-deployment.yaml
```

This deployment guide covers all major cloud platforms and deployment scenarios. Choose the approach that best fits your infrastructure requirements and scale needs.
