from collections import Counter
from flask import Flask, render_template, request, jsonify
import torch
import pickle
import re
import os
import numpy as np
from datetime import datetime
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessors
model = None
tokenizer = None
label_encoder = None
metadata = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Improved tokenizer class with handling for unknown words
class TextTokenizer:
    def __init__(self, num_words=None, oov_token='<UNK>'):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        self.document_count = 0
        self.oov_token = oov_token
        self.oov_index = 1  # Reserve index 1 for OOV
        
    def fit_on_texts(self, texts):
        """Count word frequencies and create vocabulary."""
        # First pass: count all words
        for text in texts:
            if not text or pd.isna(text):
                continue
                
            for word in text.split():
                if word:  # Skip empty strings
                    self.word_counts[word] += 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Create word_index mapping, reserving index 0 for padding
        self.word_index = {self.oov_token: self.oov_index}
        
        # Limit vocabulary size if specified
        word_list = sorted_words
        if self.num_words:
            word_list = sorted_words[:self.num_words-1]  # -1 for OOV token
            
        # Start from index 2 (0=padding, 1=OOV)
        for idx, (word, _) in enumerate(word_list, start=2):
            self.word_index[word] = idx
            
        # Create reverse mapping
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.index_word[0] = '<PAD>'
            
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices."""
        sequences = []
        for text in texts:
            if not text or pd.isna(text):
                sequences.append([])
                continue
                
            seq = []
            for word in text.split():
                # Use word index if in vocabulary, otherwise use OOV token
                if word in self.word_index:
                    seq.append(self.word_index[word])
                else:
                    seq.append(self.oov_index)
            sequences.append(seq)
        return sequences
        
    def get_vocabulary_size(self):
        """Return the size of the vocabulary including padding and OOV."""
        return len(self.word_index) + 1  # +1 for padding token

# Improved sequence padding function
def pad_sequences(sequences, maxlen=None, padding='post'):
    """Pad sequences to the same length."""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        # Handle empty sequences
        if len(seq) == 0:
            padded_seq = [0] * maxlen
            padded_sequences.append(padded_seq)
            continue
            
        # Truncate or pad the sequence
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            if padding == 'post':
                padded_seq = seq + [0] * (maxlen - len(seq))
            else:  # pre-padding
                padded_seq = [0] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)

class SpamDetectorLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(SpamDetectorLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=2,
                           batch_first=True, 
                           bidirectional=True,
                           dropout=dropout_rate)
        
        # Attention mechanism
        self.attention = torch.nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # Ensure all indices are within bounds
        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        
        # Get embeddings
        embedded = self.embedding(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout and final classification
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        
        return output

def clean_text(text):
    """Clean text for prediction."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Remove phone numbers
    text = re.sub(r'\b(?:\d{3}[-.]?){2}\d{4}\b|\+\d{1,2}\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' phone ', text)
    
    # Replace currency symbols
    text = re.sub(r'[$€£¥]', ' currency ', text)
    
    # Handle special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def pad_sequence(sequence, maxlen):
    """Pad a single sequence to fixed length."""
    if len(sequence) > maxlen:
        return sequence[:maxlen]
    else:
        return sequence + [0] * (maxlen - len(sequence))

def predict_spam(text):
    """Predict if a text is spam or ham."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    
    # Handle empty sequence
    if not sequence:
        sequence = [0]
    
    # Pad sequence
    padded = pad_sequence(sequence, metadata['max_len'])
    
    # Convert to tensor
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item() * 100
    
    # Get label
    result = label_encoder.inverse_transform([prediction])[0]
    
    return {
        'prediction': result,
        'confidence': confidence,
        'cleaned_text': cleaned_text,
        'sequence_length': len(sequence)
    }

def load_model():
    """Load the trained model and preprocessors."""
    global model, tokenizer, label_encoder, metadata
    
    try:
        # Load metadata
        with open('models/metadata.pickle', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load tokenizer
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load label encoder
        with open('models/label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Initialize model
        model = SpamDetectorLSTM(
            metadata['vocab_size'],
            metadata['embed_dim'],
            metadata['hidden_dim'],
            metadata['num_classes']
        )
        
        # Load trained weights
        checkpoint = torch.load('models/best_spam_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully with accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        return True
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text for spam detection."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    if not text.strip():
        return jsonify({'error': 'Empty text provided'}), 400
    
    try:
        result = predict_spam(text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the prediction (optional)
        with open('logs/predictions.log', 'a') as log_file:
            log_entry = f"{result['timestamp']} | {result['prediction']} | {result['confidence']:.2f}% | {text[:50]}\n"
            log_file.write(log_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend."""
    # Check if the request has JSON content
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        # Handle form data
        text = request.form.get('text', '')
        # If not in form data, try to get it from the request body
        if not text:
            text = request.data.decode('utf-8')
    
    if not text or not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = predict_spam(text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the prediction (optional)
        with open('logs/predictions.log', 'a') as log_file:
            log_entry = f"{result['timestamp']} | {result['prediction']} | {result['confidence']:.2f}% | {text[:50]}\n"
            log_file.write(log_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """API health check."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({'status': 'ok', 'message': 'Service is running'})

@app.route('/stats')
def model_stats():
    """Return model statistics."""
    if metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 503
    
    stats = {
        'accuracy': metadata.get('accuracy', 'N/A'),
        'vocabulary_size': metadata.get('vocab_size', 'N/A'),
        'embedding_dimension': metadata.get('embed_dim', 'N/A'),
        'hidden_dimension': metadata.get('hidden_dim', 'N/A'),
        'max_sequence_length': metadata.get('max_len', 'N/A')
    }
    
    return jsonify(stats)

# Create necessary directories
def setup():
    os.makedirs('logs', exist_ok=True)

if __name__ == '__main__':
    # Load the model
    with app.app_context():
        setup()


    print("Loading the spam detection model...")
    if load_model():
        print("Model loaded successfully, starting the server.")
        # Start the server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load the model. Please ensure all model files are available.")