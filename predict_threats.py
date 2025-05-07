import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import channels_data
from threat_predictor import ThreatPredictor
import sys

def format_probability(prob):
    """Format probability as percentage with color coding."""
    if prob >= 0.7:
        return f"\033[91m{prob:.1%}\033[0m"  # Red for high probability
    elif prob >= 0.4:
        return f"\033[93m{prob:.1%}\033[0m"  # Yellow for medium probability
    else:
        return f"\033[92m{prob:.1%}\033[0m"  # Green for low probability

def generate_training_labels(channels_df):
    """Generate synthetic training labels based on channel scores."""
    num_samples = len(channels_df)
    
    # Initialize labels dictionary
    labels = {
        'week': {},
        'month': {},
        'quarter': {}
    }
    
    # Generate labels for each time horizon
    for horizon in ['week', 'month', 'quarter']:
        # Generate labels for each threat category
        for category in ['ransomware_with_data_theft', 'ddos', 'cve', 'data_exfiltration', 'fraud', 'defacement']:
            # Base probability on credibility and relevance scores
            base_prob = (channels_df['Credibility Score'] + channels_df['Relevance Score']) / 20
            
            # Add some randomness
            random_factor = np.random.normal(0, 0.1, num_samples)
            prob = np.clip(base_prob + random_factor, 0, 1)
            
            # Ensure we have enough samples of each class
            min_samples_per_class = max(2, int(num_samples * 0.1))  # At least 10% of samples or 2, whichever is larger
            
            # Sort probabilities
            sorted_indices = np.argsort(prob)
            binary_labels = np.zeros(num_samples, dtype=int)
            
            # Set top probabilities to 1 (ensure at least min_samples_per_class)
            num_ones = max(min_samples_per_class, int(num_samples * 0.3))  # At least 30% positive samples
            binary_labels[sorted_indices[-num_ones:]] = 1
            
            # Ensure we have enough zeros
            num_zeros = len(binary_labels) - np.sum(binary_labels)
            if num_zeros < min_samples_per_class:
                # Convert some ones back to zeros
                ones_indices = np.where(binary_labels == 1)[0]
                num_to_convert = min_samples_per_class - num_zeros
                binary_labels[ones_indices[:num_to_convert]] = 0
            
            # Store labels
            labels[horizon][category] = binary_labels
    
    return labels

def main():
    try:
        print("\n=== Threat Probability Predictor ===\n")
        print("Loading and preparing data...")
        
        # Convert channels data to DataFrame
        channels = pd.DataFrame(channels_data.channels, columns=channels_data.columns)
        print(f"Loaded {len(channels)} channels")
        
        # Initialize predictor
        predictor = ThreatPredictor()
        predictor.prediction_threshold = 0.3
        
        # Prepare features
        print("\nPreparing features from channel data...")
        features = predictor.prepare_features(channels)
        
        # Generate training labels
        print("\nGenerating training labels...")
        labels = generate_training_labels(channels)
        
        # Train models
        print("\nTraining models...")
        predictor.train(features, labels)
        
        # Make predictions for different time horizons
        horizons = ['week', 'month', 'quarter']
        horizon_names = {
            'week': 'Next 7 Days',
            'month': 'Next 30 Days',
            'quarter': 'Next Quarter (90 Days)'
        }
        
        print("\nPredicting threats for different time horizons:")
        print("==============================================")
        
        predictions = predictor.predict(channels)
        
        # Save predictions to a JSON file
        import json
        with open('predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print("\nPredictions saved to predictions.json")
        
        # Print predictions in a readable format
        for horizon in horizons:
            print(f"\n{horizon_names[horizon]}:")
            print("-" * len(horizon_names[horizon]))
            
            for threat_type in predictor.threat_categories:
                horizon_preds = predictions[horizon][threat_type]
                avg_prob = horizon_preds['average_probability']
                peak_prob = horizon_preds['peak_probability']
                
                print(f"\n{threat_type.upper()}:")
                print(f"  Average probability: {format_probability(avg_prob)}")
                print(f"  Peak probability: {format_probability(peak_prob)}")
                
                if horizon == 'week':
                    # Show daily breakdown for week horizon
                    print("\n  Daily breakdown:")
                    for day, prob in zip(pd.date_range(start=datetime.now(), periods=7, freq='D'),
                                       horizon_preds['channel_probabilities']):
                        print(f"    {day.strftime('%Y-%m-%d')}: {format_probability(prob)}")
                
                elif horizon == 'month':
                    # Show weekly breakdown for month horizon
                    print("\n  Weekly breakdown:")
                    for week, prob in zip(pd.date_range(start=datetime.now(), periods=4, freq='W'),
                                        horizon_preds['channel_probabilities']):
                        print(f"    Week of {week.strftime('%Y-%m-%d')}: {format_probability(prob)}")
                
                elif horizon == 'quarter':
                    # Show monthly breakdown for quarter horizon
                    print("\n  Monthly breakdown:")
                    for month, prob in zip(pd.date_range(start=datetime.now(), periods=3, freq='M'),
                                         horizon_preds['channel_probabilities']):
                        print(f"    Month of {month.strftime('%Y-%m-%d')}: {format_probability(prob)}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 