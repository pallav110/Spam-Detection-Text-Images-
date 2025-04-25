from collections import Counter
from flask import Flask, render_template, request, jsonify
import torch
import pickle
import re
import os
import gc  # Added missing import
import numpy as np
from datetime import datetime
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables for model and preprocessors
model = None
image_model = None
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

def load_models():
    """Load both text and image models."""
    global model, image_model, tokenizer, label_encoder, metadata
    success = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load text model and preprocessors first
    try:
        # Load metadata
        with open('models/metadata.pickle', 'rb') as f:
            metadata = pickle.load(f)
        print("Loaded metadata")
        
        # Load tokenizer
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        print("Loaded tokenizer")
        
        # Load label encoder
        with open('models/label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Loaded label encoder")
        
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
        model = model.to(device)
        model.eval()
        
        print(f"Text model loaded successfully with accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    except Exception as e:
        print(f"Error loading text model: {e}")
        success = False
        return success
    
    # Initialize and load image model
    try:
        image_model = SpamImageClassifier()
        checkpoint = torch.load('models/spam_image_model.pt', map_location=device)
        image_model.load_state_dict(checkpoint['model_state_dict'])
        image_model = image_model.to(device)
        image_model.eval()
        print("Image model loaded successfully")
    except Exception as e:
        print(f"Error loading image model: {e}")
        success = False
    
    return success

# Image model definition
class SpamImageClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(SpamImageClassifier, self).__init__()
        # Use ResNet18 as base
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-8]:
            param.requires_grad = False
            
        # Modify the final layers for better feature extraction
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, num_classes)
        )
        
        # Add batch normalization
        self.model.avgpool = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        return self.model(x)

# Image processing imports and setup
# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_bytes):
    """Process uploaded image bytes into tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

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
    try:
        # Handle form data from textarea
        message = request.form.get('message')
        
        if not message or not message.strip():
            return jsonify({
                'error': 'No text provided',
                'prediction': 'Unknown',
                'confidence': '0.00%'
            }), 400

        # Get prediction
        result = predict_spam(message)
        
        return jsonify({
            'prediction': result['prediction'],
            'confidence': f"{result['confidence']:.2f}%",
            'text': message
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'Error processing request',
            'prediction': 'Unknown',
            'confidence': '0.00%'
        }), 500

# Image prediction route
@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    """Handle image prediction requests."""
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'prediction': 'Unknown',
                'confidence': '0.00%'
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'prediction': 'Unknown',
                'confidence': '0.00%'
            }), 400

        if not file or not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'prediction': 'Unknown',
                'confidence': '0.00%'
            }), 400

        # Process the image
        image_bytes = file.read()
        image_tensor = process_image(image_bytes)
        
        if image_tensor is None:
            return jsonify({
                'error': 'Error processing image',
                'prediction': 'Unknown',
                'confidence': '0.00%'
            }), 400

        # Move to device and predict
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = float(probabilities[0][prediction]) * 100

        result = {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'confidence': f"{confidence:.2f}%"
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error predicting image: {str(e)}")
        return jsonify({
            'error': 'Error processing image',
            'prediction': 'Unknown',
            'confidence': '0.00%'
        }), 500

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
    # Clean up any GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    print("Loading models...")
    if load_models():
        print("All models loaded successfully, starting the server.")
        # Start the server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load one or more models. Please ensure all model files are available.")