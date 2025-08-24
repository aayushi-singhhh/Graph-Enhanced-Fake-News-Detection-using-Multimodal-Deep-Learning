#!/usr/bin/env python3
"""
Comprehensive Stress Testing for Fake News Detection Pipeline
Tests with real RSS feeds and adversarial content
"""

import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import logging
from typing import List, Dict, Any
import re
from urllib.parse import urljoin
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stress_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StressTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
        # RSS Feeds from reliable sources
        self.reliable_feeds = {
            "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
            "Reuters": "http://feeds.reuters.com/reuters/topNews",
            "NYT": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "CNN": "http://rss.cnn.com/rss/edition.rss",
            "NPR": "https://feeds.npr.org/1001/rss.xml",
            "Guardian": "https://www.theguardian.com/world/rss",
            "AP": "https://storage.googleapis.com/rss-feeds/RSS-AP-Top-News.xml"
        }
        
        # Sample adversarial/suspicious content patterns
        self.adversarial_samples = [
            # Clickbait style
            "You WON'T BELIEVE What This Celebrity Did! Doctors HATE This Simple Trick!",
            
            # Conspiracy style
            "BREAKING: Government Hiding SHOCKING Truth About Vaccines - Scientists Reveal All!",
            
            # Emotional manipulation
            "URGENT: If You Don't Share This, Something TERRIBLE Will Happen to Your Family!",
            
            # Fake statistics
            "STUDY PROVES: 99.9% of People Don't Know This AMAZING Secret That Big Pharma Hides!",
            
            # WhatsApp forward style
            "Fwd: Fwd: Fwd: MUST READ!!! Bank will close your account if you don't forward this message!!!",
            
            # Misinformation patterns
            "Scientists CONFIRM: Drinking Hot Water Cures Cancer - Medical Industry Doesn't Want You to Know!",
            
            # Political manipulation
            "EXCLUSIVE LEAKED DOCUMENTS: How [Political Party] Plans to DESTROY Democracy!",
            
            # Fear mongering
            "WARNING: New Virus Spreading FAST - 100% Fatal Rate - Media Silent!!!",
            
            # Fake news with authority claims
            "Harvard Scientists Discover: This Common Food Item is Actually DEADLY Poison!",
            
            # Sensational mixed with partial truth
            "BREAKING: Major Earthquake Hits California - Death Toll Rising - Government Cover-up Exposed!"
        ]
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """Make prediction via API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"text": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def fetch_rss_articles(self, feed_url: str, max_articles: int = 10) -> List[Dict[str, str]]:
        """Fetch articles from RSS feed"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'full_text': f"{entry.get('title', '')} {entry.get('summary', '')}"
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def test_reliable_sources(self):
        """Test predictions on reliable news sources"""
        logger.info("🔍 Testing reliable news sources...")
        
        for source_name, feed_url in self.reliable_feeds.items():
            logger.info(f"Testing {source_name}...")
            articles = self.fetch_rss_articles(feed_url, max_articles=5)
            
            for article in articles:
                prediction = self.predict_text(article['full_text'])
                if prediction:
                    result = {
                        'source': source_name,
                        'type': 'reliable',
                        'title': article['title'][:100],
                        'prediction': prediction['prediction']['label'],
                        'confidence': prediction['prediction']['confidence'],
                        'expected': 'REAL',
                        'correct': prediction['prediction']['label'] == 'REAL',
                        'text_length': len(article['full_text']),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.results.append(result)
                    
                    # Log concerning predictions
                    if prediction['prediction']['label'] == 'FAKE':
                        logger.warning(f"⚠️ Reliable source {source_name} classified as FAKE: {article['title'][:50]}...")
                
                time.sleep(1)  # Rate limiting
    
    def test_adversarial_content(self):
        """Test predictions on adversarial/suspicious content"""
        logger.info("🎯 Testing adversarial content...")
        
        for i, text in enumerate(self.adversarial_samples):
            prediction = self.predict_text(text)
            if prediction:
                result = {
                    'source': f'adversarial_{i+1}',
                    'type': 'adversarial',
                    'title': text[:100],
                    'prediction': prediction['prediction']['label'],
                    'confidence': prediction['prediction']['confidence'],
                    'expected': 'FAKE',
                    'correct': prediction['prediction']['label'] == 'FAKE',
                    'text_length': len(text),
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                
                # Log concerning predictions
                if prediction['prediction']['label'] == 'REAL':
                    logger.warning(f"⚠️ Adversarial content classified as REAL: {text[:50]}...")
            
            time.sleep(0.5)
    
    def test_edge_cases(self):
        """Test edge cases and potential failure modes"""
        logger.info("🔬 Testing edge cases...")
        
        edge_cases = [
            "",  # Empty text
            "a",  # Single character
            "The.",  # Very short
            "Lorem ipsum dolor sit amet " * 100,  # Very long repetitive
            "🔥🔥🔥 AMAZING NEWS!!! 🚨🚨🚨",  # Heavy emoji usage
            "THIS IS ALL CAPS SHOUTING ABOUT FAKE NEWS",  # All caps
            "نص باللغة العربية حول الأخبار",  # Non-English (Arabic)
            "新闻测试中文文本",  # Non-English (Chinese)
            "123456789 $#@!%^&*()",  # Numbers and symbols
            "   \n\t   ",  # Only whitespace
        ]
        
        for i, text in enumerate(edge_cases):
            prediction = self.predict_text(text)
            result = {
                'source': f'edge_case_{i+1}',
                'type': 'edge_case',
                'title': repr(text)[:100],
                'prediction': prediction['prediction']['label'] if prediction else 'ERROR',
                'confidence': prediction['prediction']['confidence'] if prediction else 0.0,
                'expected': 'UNKNOWN',
                'correct': prediction is not None,  # Success if no error
                'text_length': len(text),
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
    
    def generate_report(self):
        """Generate comprehensive stress test report"""
        if not self.results:
            logger.error("No results to report!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        total_tests = len(df)
        reliable_accuracy = df[df['type'] == 'reliable']['correct'].mean() if len(df[df['type'] == 'reliable']) > 0 else 0
        adversarial_accuracy = df[df['type'] == 'adversarial']['correct'].mean() if len(df[df['type'] == 'adversarial']) > 0 else 0
        edge_case_success = df[df['type'] == 'edge_case']['correct'].mean() if len(df[df['type'] == 'edge_case']) > 0 else 0
        
        avg_confidence = df['confidence'].mean()
        fake_predictions = len(df[df['prediction'] == 'FAKE'])
        real_predictions = len(df[df['prediction'] == 'REAL'])
        
        # Generate report
        report = f"""
# 🧪 Fake News Detection Stress Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Statistics
- **Total Tests**: {total_tests}
- **Average Confidence**: {avg_confidence:.2f}
- **FAKE Predictions**: {fake_predictions} ({fake_predictions/total_tests*100:.1f}%)
- **REAL Predictions**: {real_predictions} ({real_predictions/total_tests*100:.1f}%)

## 🎯 Accuracy by Test Type
- **Reliable Sources**: {reliable_accuracy:.1%} (Should classify as REAL)
- **Adversarial Content**: {adversarial_accuracy:.1%} (Should classify as FAKE)
- **Edge Cases**: {edge_case_success:.1%} (Should handle without errors)

## 🚨 Concerning Results

### Reliable Sources Classified as FAKE:
"""
        
        # Add concerning results
        reliable_fake = df[(df['type'] == 'reliable') & (df['prediction'] == 'FAKE')]
        if len(reliable_fake) > 0:
            for _, row in reliable_fake.iterrows():
                report += f"- **{row['source']}**: {row['title']} (Confidence: {row['confidence']:.2f})\n"
        else:
            report += "- None! ✅\n"
        
        report += "\n### Adversarial Content Classified as REAL:\n"
        adversarial_real = df[(df['type'] == 'adversarial') & (df['prediction'] == 'REAL')]
        if len(adversarial_real) > 0:
            for _, row in adversarial_real.iterrows():
                report += f"- {row['title']} (Confidence: {row['confidence']:.2f})\n"
        else:
            report += "- None! ✅\n"
        
        # Confidence distribution
        report += f"""
## 📈 Confidence Distribution
- **High Confidence (>0.8)**: {len(df[df['confidence'] > 0.8])} tests
- **Medium Confidence (0.5-0.8)**: {len(df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)])} tests
- **Low Confidence (<0.5)**: {len(df[df['confidence'] < 0.5])} tests

## 💡 Recommendations
"""
        
        # Add recommendations based on results
        if reliable_accuracy < 0.8:
            report += "- ⚠️ **Reliable source accuracy is low** - Consider retraining with more diverse real news data\n"
        
        if adversarial_accuracy < 0.7:
            report += "- ⚠️ **Adversarial detection needs improvement** - Add more clickbait/fake news patterns to training\n"
        
        if avg_confidence < 0.6:
            report += "- ⚠️ **Overall confidence is low** - Model may need more training data or feature engineering\n"
        
        if edge_case_success < 0.8:
            report += "- ⚠️ **Edge case handling needs work** - Add better input validation and error handling\n"
        
        if len(df[df['confidence'] < 0.5]) / total_tests > 0.2:
            report += "- ⚠️ **Too many low-confidence predictions** - Consider ensemble methods or confidence thresholding\n"
        
        # Save report
        os.makedirs('results', exist_ok=True)
        report_file = f"results/stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save detailed results as CSV
        csv_file = f"results/stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"📄 Report saved to: {report_file}")
        logger.info(f"📊 Detailed results saved to: {csv_file}")
        
        # Print summary
        print(report)
        
        return report_file, csv_file
    
    def run_full_stress_test(self):
        """Run complete stress test suite"""
        logger.info("🚀 Starting comprehensive stress test...")
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        try:
            # Test API availability
            health_check = requests.get(f"{self.api_url}/health", timeout=10)
            if health_check.status_code != 200:
                raise Exception("API health check failed")
            
            logger.info("✅ API is available")
            
            # Run all tests
            self.test_reliable_sources()
            self.test_adversarial_content()
            self.test_edge_cases()
            
            # Generate report
            report_file, csv_file = self.generate_report()
            
            logger.info("🎉 Stress test completed successfully!")
            return report_file, csv_file
            
        except Exception as e:
            logger.error(f"❌ Stress test failed: {e}")
            raise


def main():
    """Main execution function"""
    print("🧪 Fake News Detection Stress Test")
    print("=" * 50)
    
    # Initialize tester
    tester = StressTester()
    
    try:
        # Run stress test
        report_file, csv_file = tester.run_full_stress_test()
        
        print(f"\n✅ Stress test completed!")
        print(f"📄 Report: {report_file}")
        print(f"📊 Data: {csv_file}")
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
