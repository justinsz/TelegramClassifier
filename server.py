from flask import Flask, send_file, jsonify
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/telegram_messages_classified.csv')
def serve_csv():
    return send_file('telegram_messages_classified.csv', mimetype='text/csv')

@app.route('/predictions')
def get_predictions():
    try:
        # Check if predictions file exists
        if not os.path.exists('predictions.json'):
            return jsonify({'error': 'No predictions found. Please run predict_threats.py first.'}), 404
            
        # Read the predictions
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
        
        return jsonify(predictions)
    except Exception as e:
        print(f"Error in get_predictions: {str(e)}")  # Add error logging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001) 