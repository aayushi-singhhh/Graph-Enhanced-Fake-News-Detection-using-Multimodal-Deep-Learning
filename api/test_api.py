import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the FastAPI endpoints."""
    
    print("🧪 Testing Fake News Detection API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ API not running. Start with: uvicorn api.main:app --reload")
        return
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    
    test_articles = [
        {
            "text": "Scientists at Stanford University published a peer-reviewed study showing significant improvements in cancer treatment using immunotherapy.",
            "expected": "REAL"
        },
        {
            "text": "BREAKING: World Health Organization confirms that drinking bleach cures coronavirus! Doctors don't want you to know this simple trick!",
            "expected": "FAKE"
        }
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n2.{i}. Testing article (Expected: {article['expected']})")
        print(f"Text: {article['text'][:100]}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": article["text"], "return_details": False}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                print(f"Prediction: {prediction['label']}")
                print(f"Confidence: {prediction['confidence']:.3f}")
                print(f"Probabilities: Fake={prediction['probabilities']['fake']:.3f}, Real={prediction['probabilities']['real']:.3f}")
                
                if prediction['label'] == article['expected']:
                    print("✅ Correct prediction!")
                else:
                    print("❌ Incorrect prediction")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Test simple endpoint
    print("\n3. Testing simple prediction endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict/simple",
            json={"text": "The stock market closed higher today following positive economic indicators."}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Simple prediction: {result}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test model info endpoint
    print("\n4. Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Model info: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()
