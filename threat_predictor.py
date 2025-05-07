import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator
import seaborn as sns

class ThreatPredictor:
    def __init__(self):
        """Initialize the ThreatPredictor with default settings."""
        self.threat_categories = [
            'ransomware_with_data_theft',
            'ddos',
            'cve',
            'data_exfiltration',
            'fraud',
            'defacement'
        ]
        
        self.models = {
            'week': {},
            'month': {},
            'quarter': {}
        }
        
        # Initialize models for each horizon and category
        for horizon in ['week', 'month', 'quarter']:
            for category in self.threat_categories:
                self.models[horizon][category] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                )
        
        self.prediction_threshold = 0.5
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Define time horizons in days
        self.time_horizons = {
            'week': 7,
            'month': 30,
            'quarter': 90
        }
        
        # Initialize model weights for different time horizons
        self.horizon_weights = {
            'week': 0.4,    # Higher weight for short-term predictions
            'month': 0.3,   # Medium weight for medium-term predictions
            'quarter': 0.3  # Lower weight for long-term predictions
        }
        
    def _convert_channel_list_to_dict(self, channel_data):
        """Convert channel data to dictionary format."""
        if isinstance(channel_data, dict):
            return channel_data
        elif isinstance(channel_data, (list, tuple)):
            return {
                'id': channel_data[0],
                'name': channel_data[1],
                'url': channel_data[2],
                'credibility_score': channel_data[3],
                'credibility_category': channel_data[4],
                'relevance_score': channel_data[5],
                'relevance_category': channel_data[6],
                'engagement_score': channel_data[7],
                'engagement_category': channel_data[8],
                'posting_frequency_score': channel_data[9],
                'frequency_category': channel_data[10],
                'subscriber_count_score': channel_data[11],
                'audience_category': channel_data[12]
            }
        else:
            raise ValueError("Unsupported channel data format")

    def prepare_features(self, channels_data):
        """Prepare features from channel data."""
        print("\nPreparing features:")
        print("1. Converting data format...")
        
        # Extract numerical features
        features = channels_data[[
            'Credibility Score',
            'Relevance Score',
            'Engagement Score',
            'Posting Frequency Score',
            'Subscriber Count Score'
        ]].values
        
        # Store feature names
        self.feature_names = [
            'credibility_score',
            'relevance_score',
            'engagement_score',
            'posting_frequency_score',
            'subscriber_count_score'
        ]
        
        # Add categorical features
        categorical_features = [
            'Credibility Category',
            'Relevance Category',
            'Engagement Category',
            'Frequency Category',
            'Audience Category'
        ]
        
        # Convert categorical features to one-hot encoding
        for feature in categorical_features:
            unique_values = channels_data[feature].unique()
            for value in unique_values:
                col_name = f"{feature.lower().replace(' ', '_')}_{value.lower().replace(' ', '_')}"
                self.feature_names.append(col_name)
                features = np.column_stack([
                    features,
                    (channels_data[feature] == value).astype(int)
                ])
        
        print(f"2. Processing {len(channels_data)} channels...")
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        print(f"3. Generated {features.shape[1]} features per channel")
        
        return features
    
    def _calculate_volume_features(self, group):
        """Calculate message volume-related features."""
        # Count messages per category
        label_counts = Counter()
        for labels in group['labels']:
            if isinstance(labels, list):
                label_counts.update(labels)
        
        # Calculate total volume and category percentages
        total_volume = sum(label_counts.values())
        features = {
            'total_volume': total_volume
        }
        
        # Add individual category volumes
        for category in self.threat_categories:
            features[f'{category}_volume'] = label_counts.get(category, 0)
            features[f'{category}_percentage'] = (label_counts.get(category, 0) / total_volume 
                                                if total_volume > 0 else 0)
        
        return features
    
    def _calculate_trend_features(self, df, current_date):
        """Calculate trend-related features."""
        # Look at data from the past week
        week_ago = pd.to_datetime(current_date) - timedelta(days=7)
        past_week = df[df['timestamp'].dt.date > week_ago.date()]
        
        features = {}
        # Calculate daily change rates for each category
        for category in self.threat_categories:
            daily_counts = past_week.groupby(past_week['timestamp'].dt.date).apply(
                lambda x: sum(1 for labels in x['labels'] if category in labels)
            )
            
            if len(daily_counts) > 1:
                trend = np.polyfit(range(len(daily_counts)), daily_counts, 1)[0]
                features[f'{category}_trend'] = trend
            else:
                features[f'{category}_trend'] = 0
                
        return features
    
    def _calculate_severity_features(self, group):
        """Calculate severity-related features."""
        features = {
            'avg_usefulness_score': group['usefulness_score'].mean(),
            'max_usefulness_score': group['usefulness_score'].max(),
            'critical_percentage': sum(group['criticality'] == 'CRITICAL') / len(group)
        }
        return features
    
    def _calculate_engagement_features(self, group):
        """Calculate engagement-related features."""
        features = {
            'avg_forwards': group['forwards'].mean(),
            'avg_replies': group['reply_count'].mean(),
            'max_forwards': group['forwards'].max(),
            'max_replies': group['reply_count'].max()
        }
        return features
    
    def _create_horizon_labels(self, df, current_date):
        """Create labels for different time horizons."""
        # Convert current_date to UTC timestamp
        current_date = pd.to_datetime(current_date).tz_localize(None)
        
        # Ensure df timestamp is in UTC and no timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Get future data for each horizon
        horizon_data = {}
        for horizon, days in self.time_horizons.items():
            end_date = current_date + timedelta(days=days)
            horizon_data[horizon] = df[(df['timestamp'] > current_date) & 
                                     (df['timestamp'] <= end_date)]
        
        # Create labels for each horizon
        labels = {}
        for horizon, future_data in horizon_data.items():
            horizon_labels = {}
            for category in self.threat_categories:
                # Check if this category appears in any messages
                mask = future_data['labels'].apply(lambda x: category in x if isinstance(x, list) else False)
                if mask.any():
                    # Calculate the probability based on usefulness score and frequency
                    avg_score = future_data[mask]['usefulness_score'].mean()
                    frequency = len(future_data[mask]) / len(future_data)
                    horizon_labels[category] = min(1.0, (avg_score + frequency) / 2)
                else:
                    horizon_labels[category] = 0.0
            labels[horizon] = horizon_labels
        
        return labels
    
    def train(self, features, labels):
        """Train models for each time horizon and threat category."""
        print("\nTraining models:")
        print("1. Splitting data into training and validation sets...")
        
        # Train models for each time horizon
        for horizon in ['week', 'month', 'quarter']:
            print(f"\nTraining {horizon} models:")
            for category in self.threat_categories:
                print(f"  - {category}")
                
                # Get labels for this category and horizon
                y = labels[horizon][category]
                
                # Ensure we have both classes
                if len(np.unique(y)) < 2:
                    print("    Warning: Only one class present, adding synthetic samples")
                    # Add a few synthetic samples of the missing class
                    if np.all(y == 0):
                        y[0] = 1  # Set first sample to positive
                    elif np.all(y == 1):
                        y[0] = 0  # Set first sample to negative
                
                # Split data for this specific model
                X_train, X_val, y_train, y_val = train_test_split(
                    features, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model
                model = self.models[horizon][category]
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
                
                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    F1 Score: {f1:.3f}")
        
        print("\nTraining complete!")
    
    def predict(self, channels_data):
        """Make predictions for each time horizon and threat category."""
        # Prepare features
        features = self.prepare_features(channels_data)
        
        predictions = {
            'week': {},
            'month': {},
            'quarter': {}
        }
        
        # Make predictions for each horizon and category
        for horizon in ['week', 'month', 'quarter']:
            for category in self.threat_categories:
                model = self.models[horizon][category]
                
                # Get probabilities for positive class
                probabilities = model.predict_proba(features)[:, 1]
                
                # Calculate average and peak probabilities
                avg_prob = float(np.mean(probabilities))
                peak_prob = float(np.max(probabilities))
                
                # Store predictions
                predictions[horizon][category] = {
                    'average_probability': avg_prob,
                    'peak_probability': peak_prob,
                    'channel_probabilities': probabilities.tolist()
                }
        
        return predictions
    
    def evaluate(self, test_channels, test_labels):
        """
        Evaluate model performance on test data.
        Returns evaluation metrics for each time horizon.
        """
        predictions = self.predict(test_channels)
        
        # Extract predictions for each horizon
        week_preds = [p['predictions']['week']['is_threat'] for p in predictions]
        month_preds = [p['predictions']['month']['is_threat'] for p in predictions]
        quarter_preds = [p['predictions']['quarter']['is_threat'] for p in predictions]
        
        # Calculate metrics for each horizon
        metrics = {}
        horizons = ['week', 'month', 'quarter']
        preds = [week_preds, month_preds, quarter_preds]
        
        for horizon, pred in zip(horizons, preds):
            metrics[horizon] = {
                'accuracy': accuracy_score(test_labels[horizon], pred),
                'precision': precision_score(test_labels[horizon], pred),
                'recall': recall_score(test_labels[horizon], pred),
                'f1': f1_score(test_labels[horizon], pred)
            }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'week_model': self.week_model,
            'month_model': self.month_model,
            'quarter_model': self.quarter_model,
            'scaler': self.scaler,
            'threat_categories': self.threat_categories,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.week_model = model_data['week_model']
        self.month_model = model_data['month_model']
        self.quarter_model = model_data['quarter_model']
        self.scaler = model_data['scaler']
        self.threat_categories = model_data['threat_categories']
        self.feature_names = model_data['feature_names']
    
    def visualize_threat_trends(self, df, output_dir='visualizations'):
        """Generate visualizations of threat trends."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Processing data...")
        # Convert timestamp and ensure no timezone issues
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df['date'] = df['timestamp'].dt.date
        
        # Filter for last 30 days
        last_date = df['date'].max()
        start_date = last_date - timedelta(days=30)
        df_recent = df[df['date'] > start_date]
        print(f"Filtering data to last 30 days: {start_date} to {last_date}")
        
        # Extract threat categories if not already set
        if not self.threat_categories:
            all_labels = []
            for labels in df_recent['labels']:
                if isinstance(labels, list) and labels:
                    all_labels.extend(labels)
            self.threat_categories = list(set(all_labels))
        
        if not self.threat_categories:
            print("Warning: No threat categories found in the data")
            return "No visualizations generated - no threat categories found"
            
        print(f"Found {len(self.threat_categories)} threat categories")
        
        # 1. Daily threat volume by category
        print("Generating daily threat volume plot...")
        plt.figure(figsize=(15, 8))
        daily_threats = {}
        for category in self.threat_categories:
            # Fix deprecation warning by explicitly selecting columns
            daily_counts = df_recent.groupby('date')['labels'].apply(
                lambda x: sum(1 for labels in x if isinstance(labels, list) and category in labels)
            )
            if not daily_counts.empty:
                plt.plot(pd.to_datetime(daily_counts.index), daily_counts.values, 
                        label=category, alpha=0.7)
        
        plt.title('Daily Threat Volume by Category (Last 30 Days)', pad=20, fontsize=14)
        plt.xlabel('Date', labelpad=10, fontsize=12)
        plt.ylabel('Number of Threats', labelpad=10, fontsize=12)
        
        # Format x-axis with proper date locators for 30-day view
        ax = plt.gca()
        ax.xaxis.set_major_locator(DayLocator(interval=5))  # Show every 5 days
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Format y-axis
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.grid(True, which='major', linestyle='-', alpha=0.3)
        
        # Add legend with better positioning and formatting
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f'{output_dir}/daily_threat_volume.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Threat severity distribution
        print("Generating severity distribution plot...")
        plt.figure(figsize=(12, 6))
        if 'criticality' in df_recent.columns and not df_recent['criticality'].isna().all():
            severity_counts = df_recent['criticality'].value_counts()
            sns.barplot(x=severity_counts.index, y=severity_counts.values)
            plt.title('Distribution of Threat Severity Levels')
            plt.xlabel('Severity Level')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/severity_distribution.png')
        plt.close()
        
        # 3. Engagement correlation heatmap
        print("Generating engagement correlation plot...")
        plt.figure(figsize=(10, 8))
        engagement_cols = ['forwards', 'reply_count', 'usefulness_score']
        valid_cols = [col for col in engagement_cols if col in df_recent.columns and not df_recent[col].isna().all()]
        if valid_cols:
            correlation = df_recent[valid_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('Engagement Metrics Correlation')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/engagement_correlation.png')
        plt.close()
        
        # 4. Moving average of threat scores
        print("Generating threat score moving average plot...")
        plt.figure(figsize=(15, 8))
        df_sorted = df_recent.sort_values('timestamp')
        window_size = 7  # 7-day moving average
        
        has_data = False
        for category in self.threat_categories:
            # Fix deprecation warning by explicitly selecting columns
            daily_scores = df_sorted.groupby('date')['usefulness_score'].apply(
                lambda x: x[df_sorted[df_sorted['date'] == x.name]['labels'].apply(
                    lambda y: isinstance(y, list) and category in y)].mean()
            )
            if not daily_scores.isna().all():
                has_data = True
                daily_scores = daily_scores.rolling(window=window_size, min_periods=1).mean()
                plt.plot(pd.to_datetime(daily_scores.index), daily_scores.values, 
                        label=category, alpha=0.7)
        
        if has_data:
            plt.title(f'{window_size}-Day Moving Average of Threat Scores by Category',
                     pad=20, fontsize=14)
            plt.xlabel('Date', labelpad=10, fontsize=12)
            plt.ylabel('Average Threat Score', labelpad=10, fontsize=12)
            
            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(MonthLocator())
            
            # Add grid
            plt.grid(True, which='major', linestyle='-', alpha=0.3)
            plt.grid(True, which='minor', linestyle=':', alpha=0.1)
            
            # Add legend with better formatting
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
        plt.savefig(f'{output_dir}/threat_score_moving_average.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Threat category co-occurrence matrix
        print("Generating threat co-occurrence matrix...")
        plt.figure(figsize=(12, 10))
        cooccurrence = np.zeros((len(self.threat_categories), len(self.threat_categories)))
        
        has_cooccurrence = False
        for labels in df_recent['labels']:
            if isinstance(labels, list) and len(labels) > 1:
                has_cooccurrence = True
                for i, cat1 in enumerate(self.threat_categories):
                    for j, cat2 in enumerate(self.threat_categories):
                        if cat1 in labels and cat2 in labels:
                            cooccurrence[i, j] += 1
        
        if has_cooccurrence:
            sns.heatmap(cooccurrence, 
                       xticklabels=self.threat_categories,
                       yticklabels=self.threat_categories,
                       annot=True, fmt='g', cmap='YlOrRd')
            plt.title('Threat Category Co-occurrence Matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/threat_cooccurrence.png')
        plt.close()
        
        print("Visualization generation complete!")
        return f"Visualizations saved to {output_dir}/"

    def set_prediction_threshold(self, threshold):
        """Set the prediction threshold for converting probabilities to binary predictions.
        
        Args:
            threshold (float): Value between 0 and 1 to use as prediction threshold
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.prediction_threshold = threshold

    def predict_proba(self, features):
        """Predict threat probabilities for each horizon.
        
        Args:
            features (pd.DataFrame): Feature matrix
            
        Returns:
            dict: Dictionary with predictions for each horizon
                  Each prediction is a numpy array of shape (n_samples, n_categories)
        """
        if not hasattr(self, 'models'):
            raise ValueError("Models not trained. Call train() first.")
            
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        predictions = {}
        
        # Get probability predictions for each horizon
        for horizon in ['week', 'month', 'quarter']:
            # Initialize probability matrix for this horizon
            horizon_probs = np.zeros((len(features), len(self.threat_categories)))
            
            # Get probabilities for each category
            for i, category in enumerate(self.threat_categories):
                # For binary classification, we only need the probability of class 1
                probs = self.models[horizon][category].predict(scaled_features).astype(float)
                horizon_probs[:, i] = probs
            
            predictions[horizon] = horizon_probs
        
        return predictions

def main():
    print("\n=== Telegram Threat Predictor ===\n")
    
    # Load classified data
    print("Step 1/5: Loading and preprocessing data...")
    df = pd.read_csv('telegram_messages_classified.csv')
    print(f"Loaded {len(df)} messages")
    
    # Convert string representations of lists to actual lists
    print("Converting data types...")
    df['labels'] = df['labels'].apply(eval)
    df['matching_keywords'] = df['matching_keywords'].apply(eval)
    
    # Initialize predictor
    predictor = ThreatPredictor()
    
    # Generate visualizations
    print("\nStep 2/5: Generating threat trend visualizations...")
    viz_path = predictor.visualize_threat_trends(df)
    print(f"Visualizations saved to: {viz_path}")
    
    # Prepare features and labels
    print("\nStep 3/5: Preparing features and labels...")
    features, labels = predictor.prepare_features(df)
    print(f"Generated {features.shape[1]} features for training")
    
    # Split data
    print("\nStep 4/5: Splitting training and test data...")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(train_features)} samples")
    print(f"Test set size: {len(test_features)} samples")
    
    # Train model
    print("\nStep 5/5: Training prediction models...")
    predictor.train(train_features, train_labels)
    
    # Set optimal prediction configuration
    print("\nApplying optimal prediction configuration...")
    predictor.set_prediction_threshold(0.3)  # Using recommended threshold from testing
    
    # Evaluate model using predict method
    print("\nEvaluating model performance...")
    results = predictor.evaluate(test_features, test_labels)
    
    # Print results
    print("\n=== Threat Prediction Results ===")
    for horizon, metrics in results.items():
        print(f"\n{horizon.title()} Horizon Predictions:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1']:.2f}")
    
    # Make predictions on recent data
    print("\nGenerating predictions for recent data...")
    recent_features = features.iloc[-5:]  # Get last 5 days of data
    predictions = predictor.predict(recent_features)
    
    print("\nRecent Threat Predictions:")
    for horizon in ['week', 'month', 'quarter']:
        print(f"\n{horizon.title()} Horizon Forecast:")
        for i, pred in enumerate(predictions[horizon], 1):
            print(f"Day {i}: {pred}")
    
    # Save model
    print("\nSaving trained model...")
    predictor.save_model('threat_predictor_model.joblib')
    print("Model saved as: threat_predictor_model.joblib")
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main() 