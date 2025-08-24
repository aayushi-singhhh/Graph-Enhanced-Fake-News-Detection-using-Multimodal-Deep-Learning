import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import time
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Monitoring and alerting
import logging
from typing import Dict, List, Any, Optional

class ModelMonitor:
    """
    Production monitoring system for fake news detection model.
    Tracks model performance, data drift, and prediction patterns.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None, log_dir="logs"):
        """Initialize model monitor."""
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.log_dir = log_dir
        
        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Monitoring metrics
        self.metrics_history = defaultdict(list)
        self.prediction_log = []
        self.drift_alerts = []
        
        # Thresholds for alerts
        self.thresholds = {
            'accuracy_drop': 0.05,  # 5% drop in accuracy
            'confidence_drop': 0.1,  # 10% drop in confidence
            'fake_ratio_change': 0.2,  # 20% change in fake news ratio
            'response_time': 5.0  # 5 seconds max response time
        }
        
        # Load baseline metrics
        self.load_baseline_metrics()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.log_dir, f"model_monitor_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model Monitor initialized")
    
    def load_baseline_metrics(self):
        """Load baseline performance metrics."""
        try:
            baseline_file = os.path.join(self.log_dir, "baseline_metrics.json")
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    self.baseline_metrics = json.load(f)
                self.logger.info("Baseline metrics loaded")
            else:
                # Default baseline metrics (update with actual training results)
                self.baseline_metrics = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.79,
                    'f1': 0.80,
                    'avg_confidence': 0.75,
                    'fake_ratio': 0.15,
                    'avg_response_time': 0.5
                }
                self.save_baseline_metrics()
                self.logger.info("Default baseline metrics created")
        
        except Exception as e:
            self.logger.error(f"Error loading baseline metrics: {str(e)}")
            self.baseline_metrics = {}
    
    def save_baseline_metrics(self):
        """Save baseline metrics to file."""
        try:
            baseline_file = os.path.join(self.log_dir, "baseline_metrics.json")
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_metrics, f, indent=2)
            self.logger.info("Baseline metrics saved")
        except Exception as e:
            self.logger.error(f"Error saving baseline metrics: {str(e)}")
    
    def log_prediction(self, text: str, prediction: Dict[str, Any], response_time: float):
        """Log individual prediction for monitoring."""
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'word_count': len(text.split()),
            'prediction': prediction['label'],
            'confidence': prediction['confidence'],
            'fake_probability': prediction['probabilities']['fake'],
            'real_probability': prediction['probabilities']['real'],
            'response_time': response_time
        }
        
        self.prediction_log.append(prediction_entry)
        
        # Keep only last 1000 predictions in memory
        if len(self.prediction_log) > 1000:
            self.prediction_log = self.prediction_log[-1000:]
        
        # Check for alerts
        self.check_real_time_alerts(prediction_entry)
    
    def check_real_time_alerts(self, prediction_entry: Dict[str, Any]):
        """Check for real-time alerts based on current prediction."""
        # Response time alert
        if prediction_entry['response_time'] > self.thresholds['response_time']:
            alert = {
                'type': 'response_time',
                'message': f"High response time: {prediction_entry['response_time']:.2f}s",
                'timestamp': prediction_entry['timestamp'],
                'severity': 'warning'
            }
            self.trigger_alert(alert)
        
        # Very low confidence alert
        if prediction_entry['confidence'] < 0.5:
            alert = {
                'type': 'low_confidence',
                'message': f"Very low confidence prediction: {prediction_entry['confidence']:.3f}",
                'timestamp': prediction_entry['timestamp'],
                'severity': 'info'
            }
            self.trigger_alert(alert)
    
    def analyze_recent_performance(self, hours: int = 24) -> Dict[str, float]:
        """Analyze model performance over recent time period."""
        if not self.prediction_log:
            return {}
        
        # Filter recent predictions
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = [
            p for p in self.prediction_log 
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
        if not recent_predictions:
            return {}
        
        # Calculate metrics
        df = pd.DataFrame(recent_predictions)
        
        metrics = {
            'total_predictions': len(recent_predictions),
            'avg_confidence': df['confidence'].mean(),
            'fake_ratio': (df['prediction'] == 'FAKE').mean(),
            'avg_response_time': df['response_time'].mean(),
            'low_confidence_ratio': (df['confidence'] < 0.6).mean(),
            'high_confidence_ratio': (df['confidence'] > 0.8).mean()
        }
        
        return metrics
    
    def detect_data_drift(self, hours: int = 24) -> Dict[str, Any]:
        """Detect potential data drift in recent predictions."""
        recent_metrics = self.analyze_recent_performance(hours)
        
        if not recent_metrics or not self.baseline_metrics:
            return {'drift_detected': False, 'alerts': []}
        
        drift_alerts = []
        
        # Check confidence drift
        if 'avg_confidence' in self.baseline_metrics:
            confidence_change = abs(recent_metrics['avg_confidence'] - self.baseline_metrics['avg_confidence'])
            if confidence_change > self.thresholds['confidence_drop']:
                drift_alerts.append({
                    'metric': 'confidence',
                    'baseline': self.baseline_metrics['avg_confidence'],
                    'current': recent_metrics['avg_confidence'],
                    'change': confidence_change,
                    'severity': 'warning' if confidence_change > 0.15 else 'info'
                })
        
        # Check fake ratio drift
        if 'fake_ratio' in self.baseline_metrics:
            ratio_change = abs(recent_metrics['fake_ratio'] - self.baseline_metrics['fake_ratio'])
            if ratio_change > self.thresholds['fake_ratio_change']:
                drift_alerts.append({
                    'metric': 'fake_ratio',
                    'baseline': self.baseline_metrics['fake_ratio'],
                    'current': recent_metrics['fake_ratio'],
                    'change': ratio_change,
                    'severity': 'warning'
                })
        
        # Check response time drift
        if 'avg_response_time' in self.baseline_metrics:
            time_change = recent_metrics['avg_response_time'] - self.baseline_metrics['avg_response_time']
            if time_change > 1.0:  # 1 second increase
                drift_alerts.append({
                    'metric': 'response_time',
                    'baseline': self.baseline_metrics['avg_response_time'],
                    'current': recent_metrics['avg_response_time'],
                    'change': time_change,
                    'severity': 'warning'
                })
        
        return {
            'drift_detected': len(drift_alerts) > 0,
            'alerts': drift_alerts,
            'recent_metrics': recent_metrics
        }
    
    def trigger_alert(self, alert: Dict[str, Any]):
        """Trigger and log an alert."""
        self.drift_alerts.append(alert)
        
        # Log alert
        self.logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        # Keep only last 100 alerts
        if len(self.drift_alerts) > 100:
            self.drift_alerts = self.drift_alerts[-100:]
        
        # In production, you could send notifications here:
        # - Email alerts
        # - Slack notifications  
        # - Dashboard updates
        # - PagerDuty integration
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""
        report = []
        report.append("🔍 MODEL MONITORING REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Recent performance
        recent_24h = self.analyze_recent_performance(24)
        recent_1h = self.analyze_recent_performance(1)
        
        if recent_24h:
            report.append(f"\n📊 Performance (Last 24 Hours):")
            report.append(f"   - Total Predictions: {recent_24h['total_predictions']:,}")
            report.append(f"   - Average Confidence: {recent_24h['avg_confidence']:.3f}")
            report.append(f"   - Fake News Ratio: {recent_24h['fake_ratio']:.3f}")
            report.append(f"   - Average Response Time: {recent_24h['avg_response_time']:.3f}s")
            report.append(f"   - Low Confidence Ratio: {recent_24h['low_confidence_ratio']:.3f}")
        
        if recent_1h:
            report.append(f"\n📈 Performance (Last Hour):")
            report.append(f"   - Total Predictions: {recent_1h['total_predictions']:,}")
            report.append(f"   - Average Confidence: {recent_1h['avg_confidence']:.3f}")
            report.append(f"   - Fake News Ratio: {recent_1h['fake_ratio']:.3f}")
        
        # Drift detection
        drift_results = self.detect_data_drift()
        if drift_results['drift_detected']:
            report.append(f"\n⚠️  DRIFT ALERTS:")
            for alert in drift_results['alerts']:
                change_direction = "↑" if alert['change'] > 0 else "↓"
                report.append(f"   - {alert['metric'].upper()}: {alert['baseline']:.3f} → {alert['current']:.3f} {change_direction}")
        else:
            report.append(f"\n✅ No significant drift detected")
        
        # Recent alerts
        recent_alerts = [a for a in self.drift_alerts if 
                        datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        if recent_alerts:
            report.append(f"\n🚨 Recent Alerts (24h): {len(recent_alerts)}")
            for alert in recent_alerts[-5:]:  # Show last 5
                report.append(f"   - [{alert['type']}] {alert['message']}")
        else:
            report.append(f"\n✅ No recent alerts")
        
        # Baseline comparison
        if self.baseline_metrics and recent_24h:
            report.append(f"\n📊 Baseline Comparison:")
            for metric in ['avg_confidence', 'fake_ratio', 'avg_response_time']:
                if metric in self.baseline_metrics and metric in recent_24h:
                    baseline = self.baseline_metrics[metric]
                    current = recent_24h[metric]
                    change = current - baseline
                    change_pct = (change / baseline) * 100 if baseline != 0 else 0
                    direction = "↑" if change > 0 else "↓"
                    report.append(f"   - {metric}: {baseline:.3f} → {current:.3f} ({change_pct:+.1f}%) {direction}")
        
        return "\n".join(report)
    
    def save_monitoring_data(self):
        """Save monitoring data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prediction log
        if self.prediction_log:
            log_file = os.path.join(self.log_dir, f"predictions_{timestamp}.json")
            with open(log_file, 'w') as f:
                json.dump(self.prediction_log, f, indent=2)
        
        # Save alerts
        if self.drift_alerts:
            alerts_file = os.path.join(self.log_dir, f"alerts_{timestamp}.json")
            with open(alerts_file, 'w') as f:
                json.dump(self.drift_alerts, f, indent=2)
        
        # Save current metrics
        current_metrics = self.analyze_recent_performance(24)
        if current_metrics:
            metrics_file = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(current_metrics, f, indent=2)
    
    def create_monitoring_dashboard(self):
        """Create visual monitoring dashboard."""
        if not self.prediction_log:
            print("📊 No prediction data available for dashboard")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.prediction_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fake News Detection Model - Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Confidence distribution
        axes[0, 0].hist(df['confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["confidence"].mean():.3f}')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Prediction Confidence Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Predictions over time
        df_hourly = df.set_index('timestamp').resample('H').agg({
            'prediction': lambda x: (x == 'FAKE').sum(),
            'confidence': 'mean'
        })
        
        ax2 = axes[0, 1]
        ax2.plot(df_hourly.index, df_hourly['prediction'], marker='o', color='red', alpha=0.7, label='Fake Count')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fake News Count (per hour)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_title('Fake News Predictions Over Time')
        ax2.grid(alpha=0.3)
        
        # Secondary y-axis for confidence
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df_hourly.index, df_hourly['confidence'], marker='s', color='blue', alpha=0.7, label='Avg Confidence')
        ax2_twin.set_ylabel('Average Confidence', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        
        # 3. Response time analysis
        axes[1, 0].scatter(df['word_count'], df['response_time'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('Response Time (seconds)')
        axes[1, 0].set_title('Response Time vs Text Length')
        axes[1, 0].grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['word_count'], df['response_time'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(df['word_count'], p(df['word_count']), "r--", alpha=0.8)
        
        # 4. Prediction summary
        prediction_counts = df['prediction'].value_counts()
        colors = ['lightcoral', 'lightgreen']
        axes[1, 1].pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[1, 1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = os.path.join(self.log_dir, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved to: {dashboard_path}")
        
        plt.show()

class RetrainingScheduler:
    """
    Automated retraining scheduler for the fake news detection model.
    """
    
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.retraining_history = []
    
    def should_retrain(self) -> Dict[str, Any]:
        """Determine if model should be retrained based on drift and performance."""
        drift_results = self.monitor.detect_data_drift()
        recent_metrics = self.monitor.analyze_recent_performance(24)
        
        retrain_reasons = []
        
        # Check for significant drift
        if drift_results['drift_detected']:
            high_severity_alerts = [a for a in drift_results['alerts'] if a['severity'] == 'warning']
            if high_severity_alerts:
                retrain_reasons.append("Significant data drift detected")
        
        # Check prediction volume
        if recent_metrics and recent_metrics['total_predictions'] < 10:
            retrain_reasons.append("Low prediction volume - insufficient recent data")
        
        # Check confidence drop
        if (recent_metrics and self.monitor.baseline_metrics and 
            'avg_confidence' in self.monitor.baseline_metrics):
            confidence_drop = (self.monitor.baseline_metrics['avg_confidence'] - 
                             recent_metrics['avg_confidence'])
            if confidence_drop > self.monitor.thresholds['confidence_drop']:
                retrain_reasons.append(f"Confidence drop: {confidence_drop:.3f}")
        
        return {
            'should_retrain': len(retrain_reasons) > 0,
            'reasons': retrain_reasons,
            'drift_results': drift_results,
            'recent_metrics': recent_metrics
        }
    
    def schedule_retraining(self):
        """Schedule and log retraining recommendation."""
        retrain_decision = self.should_retrain()
        
        if retrain_decision['should_retrain']:
            retraining_entry = {
                'timestamp': datetime.now().isoformat(),
                'reasons': retrain_decision['reasons'],
                'drift_alerts': retrain_decision['drift_results']['alerts'],
                'recent_metrics': retrain_decision['recent_metrics'],
                'status': 'recommended'
            }
            
            self.retraining_history.append(retraining_entry)
            
            self.monitor.logger.warning(f"RETRAINING RECOMMENDED: {', '.join(retrain_decision['reasons'])}")
            
            return retraining_entry
        
        return None

def main():
    """Main function for monitoring demonstration."""
    
    print("📊 Model Monitoring System Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = ModelMonitor()
    
    # Simulate some predictions for demonstration
    print("🔄 Simulating predictions for monitoring demo...")
    
    sample_predictions = [
        {"text": "Scientists discover new cancer treatment", "label": "REAL", "confidence": 0.85, "probabilities": {"fake": 0.15, "real": 0.85}},
        {"text": "SHOCKING: Aliens land in NYC!", "label": "FAKE", "confidence": 0.92, "probabilities": {"fake": 0.92, "real": 0.08}},
        {"text": "Stock market rises on economic news", "label": "REAL", "confidence": 0.78, "probabilities": {"fake": 0.22, "real": 0.78}},
        {"text": "Miracle cure doctors hate!", "label": "FAKE", "confidence": 0.88, "probabilities": {"fake": 0.88, "real": 0.12}},
        {"text": "Weather forecast shows rain tomorrow", "label": "REAL", "confidence": 0.65, "probabilities": {"fake": 0.35, "real": 0.65}},
    ]
    
    # Log simulated predictions
    for i, pred in enumerate(sample_predictions):
        response_time = np.random.uniform(0.1, 2.0)  # Random response time
        monitor.log_prediction(pred["text"], pred, response_time)
        time.sleep(0.1)  # Small delay for demo
    
    print(f"✅ Logged {len(sample_predictions)} predictions")
    
    # Generate monitoring report
    print("\n📋 Generating monitoring report...")
    report = monitor.generate_monitoring_report()
    print(report)
    
    # Check for retraining
    print("\n🔄 Checking retraining recommendations...")
    scheduler = RetrainingScheduler(monitor)
    retrain_recommendation = scheduler.schedule_retraining()
    
    if retrain_recommendation:
        print("⚠️  Retraining recommended!")
        for reason in retrain_recommendation['reasons']:
            print(f"   - {reason}")
    else:
        print("✅ No retraining needed at this time")
    
    # Create dashboard
    print("\n📊 Creating monitoring dashboard...")
    monitor.create_monitoring_dashboard()
    
    # Save monitoring data
    print("\n💾 Saving monitoring data...")
    monitor.save_monitoring_data()
    
    print("\n✅ Monitoring demonstration completed!")

if __name__ == "__main__":
    main()
