# SpamShield - Advanced Spam Detection System

SpamShield is a sophisticated spam detection system that uses deep learning to analyze both text and images for spam content. The system employs state-of-the-art models including LSTM for text analysis and ResNet18 for image classification.

## Features

- **Dual-Mode Analysis**: Detect spam in both text content and images
- **Real-time Processing**: Instant analysis with confidence scores
- **Modern UI**: Responsive design with dark/light mode support
- **Advanced ML Models**:
  - Text Analysis: Bidirectional LSTM with attention mechanism
  - Image Analysis: Fine-tuned ResNet18 with custom classification layers

## Technical Architecture

### Text Classification Model
- Bidirectional LSTM with attention mechanism
- Advanced text preprocessing and tokenization
- Handles class imbalance through weighted sampling
- Early stopping and learning rate scheduling

### Image Classification Model
- Pre-trained ResNet18 backbone
- Custom classification layers with dropout
- Batch normalization for training stability
- Data augmentation for better generalization

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the following directory structure:
```
SPAM-DETECTION/
├── models/              # Pre-trained model files
├── data/               # Dataset directory
│   └── Images/         # Image dataset
├── logs/               # Prediction logs
├── uploads/            # Temporary upload directory
├── visualizations/     # Training visualizations
└── templates/          # HTML templates
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Choose between text or image analysis:
   - For text: Enter or paste the content to analyze
   - For images: Upload or drag & drop an image file

## Model Training

### Text Model Training
```bash
python spam_detector.py
```

### Image Model Training
```bash
python image_classifier.py
```

## API Endpoints

- `POST /predict`: Text spam detection
- `POST /predict_image`: Image spam detection
- `GET /health`: Health check
- `GET /stats`: Model statistics

## Performance

The system achieves:
- Text Classification: ~98% accuracy on test set
- Image Classification: ~95% accuracy on test set
- Real-time processing: <2s response time

## Dependencies

See requirements.txt for full list of dependencies.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.