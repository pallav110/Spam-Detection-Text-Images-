import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from collections import Counter
import re
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
from multiprocessing import freeze_support

# Enhanced text cleaning function
def clean_text(text):
    """Advanced text cleaning for spam detection."""
    if text is None or pd.isna(text) or not isinstance(text, str):
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

# Visualize dataset characteristics
def visualize_data_distribution(df, label_column, save_path=None):
    """Create visualizations of data distribution."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=label_column, data=df)
    plt.title('Distribution of Ham vs Spam Messages')
    plt.xlabel('Message Type')
    plt.ylabel('Count')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

# Text length analysis
def visualize_text_length(df, text_column, label_column, save_path=None):
    """Visualize the distribution of text lengths by label."""
    df['text_length'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue=label_column, kde=True, bins=50)
    plt.title('Distribution of Message Lengths')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.xlim(0, df['text_length'].quantile(0.99))  # Limit to 99th percentile for better visualization
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()
    
    return df

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

# Define an improved LSTM model with attention mechanism
class SpamDetectorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(SpamDetectorLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=2,  # Two layers for better learning
                           batch_first=True, 
                           bidirectional=True,  # Bidirectional for better context
                           dropout=dropout_rate)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # Ensure all indices are within bounds
        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        
        # Get embeddings
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout and final classification
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        
        return output

# Main function
if __name__ == '__main__':
    # For Windows multiprocessing support
    freeze_support()
    
    # Setup directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Memory optimization: Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Check CUDA availability and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB')
        print(f'Memory Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB')

    # Data loading and preprocessing
    print("Loading and preprocessing data...")
    try:
        # Try to load the dataset from the specified location
        data_path = 'data/spam-detection/enron_spam_data.csv'
        if not os.path.exists(data_path):
            data_path = 'data/enron_spam_data.csv'
        
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Check for ham/spam column
        label_col = None
        text_col = None
        
        # Try to automatically identify the column names
        possible_label_cols = ['Spam/Ham', 'spam', 'label', 'class', 'Category']
        possible_text_cols = ['Message', 'text', 'content', 'message', 'email', 'sms']
        
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if label_col is None or text_col is None:
            print("Column names not automatically recognized. Please specify them:")
            print(f"Available columns: {df.columns.tolist()}")
            label_col = input("Enter label column name: ")
            text_col = input("Enter text column name: ")
        
        print(f"Using '{label_col}' as label column and '{text_col}' as text column.")
        
        # Data cleanup - Drop rows with missing values
        original_size = len(df)
        df = df.dropna(subset=[text_col, label_col])
        print(f"Dropped {original_size - len(df)} rows with missing values.")
        
        # Standardize label values
        # Map different variations of ham/spam to standard format
        label_mapping = {
            'ham': 'ham', 'Ham': 'ham', '0': 'ham', 0: 'ham',
            'spam': 'spam', 'Spam': 'spam', '1': 'spam', 1: 'spam'
        }
        
        # Apply mapping, drop rows with invalid labels
        df[label_col] = df[label_col].map(label_mapping)
        df = df[df[label_col].isin(['ham', 'spam'])]
        print(f"After filtering invalid labels: {len(df)} rows")
        
        # Visualize data distribution before resampling
        visualize_data_distribution(df, label_col, 'visualizations/before_resampling.png')
        
        # Handle class imbalance through resampling
        ham_df = df[df[label_col] == 'ham']
        spam_df = df[df[label_col] == 'spam']
        
        print(f"Before resampling - Ham: {len(ham_df)}, Spam: {len(spam_df)}")
        
        # Resample data to balance classes
        if len(ham_df) > len(spam_df):
            # Downsample ham class
            ham_downsampled = resample(ham_df, 
                                      replace=False,
                                      n_samples=len(spam_df),
                                      random_state=42)
            balanced_df = pd.concat([ham_downsampled, spam_df])
        else:
            # Downsample spam class
            spam_downsampled = resample(spam_df, 
                                       replace=False,
                                       n_samples=len(ham_df),
                                       random_state=42)
            balanced_df = pd.concat([ham_df, spam_downsampled])
        
        print(f"After resampling - Total: {len(balanced_df)}")
        
        # Visualize balanced dataset
        visualize_data_distribution(balanced_df, label_col, 'visualizations/after_resampling.png')
        
        # Clean text data
        print("Cleaning text data...")
        balanced_df[text_col] = balanced_df[text_col].apply(clean_text)
        
        # Remove rows with empty text after cleaning
        balanced_df = balanced_df[balanced_df[text_col].str.strip() != '']
        print(f"After removing empty texts: {len(balanced_df)} rows")
        
        # Analyze text length
        balanced_df = visualize_text_length(balanced_df, text_col, label_col, 'visualizations/text_length.png')
        
        # Save processed dataset
        balanced_df.to_csv('data/processed/clean_balanced_data.csv', index=False)
        print("Saved clean balanced dataset.")
        
        # Convert labels to numerical form
        le = LabelEncoder()
        Y = le.fit_transform(balanced_df[label_col])
        Y = torch.LongTensor(Y)
        
        # Tokenization
        print("Tokenizing text data...")
        max_features = 10000  # Vocabulary size
        tokenizer = TextTokenizer(num_words=max_features)
        tokenizer.fit_on_texts(balanced_df[text_col])
        
        # Convert texts to sequences
        X_sequences = tokenizer.texts_to_sequences(balanced_df[text_col])
        
        # Calculate sequence statistics
        seq_lengths = [len(seq) for seq in X_sequences]
        print(f"Sequence length stats - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Avg: {sum(seq_lengths)/len(seq_lengths):.2f}")
        
        # Display common vocabulary words
        print("\nTop 10 most common words:")
        for word, count in tokenizer.word_counts.most_common(10):
            print(f"  {word}: {count}")
            
        # Define maximum sequence length - use 95th percentile to avoid outliers
        max_len = int(np.percentile(seq_lengths, 95))
        print(f"Setting maximum sequence length to: {max_len}")
        
        # Pad sequences
        X = pad_sequences(X_sequences, maxlen=max_len)
        print(f"Input tensor shape: {X.shape}")
        
        # Split data with stratification
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        
        # Create DataLoaders
        batch_size = 64
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0  # Safer setting for CUDA
        )

        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )
        
        # Initialize model
        vocab_size = tokenizer.get_vocabulary_size()
        embed_dim = 128
        hidden_dim = 128
        num_classes = 2
        model = SpamDetectorLSTM(vocab_size, embed_dim, hidden_dim, num_classes)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop with early stopping
        num_epochs = 10
        patience = 3  # Number of epochs to wait for improvement
        best_val_loss = float('inf')
        no_improve_epochs = 0
        train_losses = []
        val_losses = []
        
        print("Starting training...")
        
        try:
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for batch_x, batch_y in progress_bar:
                    # Move batch to device
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update progress
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Memory management
                    del outputs, loss
                    batch_x = batch_x.cpu()
                    batch_y = batch_y.cpu()
                
                avg_train_loss = total_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in tqdm(test_loader, desc='Validation'):
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                        # Memory management
                        batch_x = batch_x.cpu()
                        batch_y = batch_y.cpu()
                
                avg_val_loss = val_loss / len(test_loader)
                val_accuracy = 100 * val_correct / val_total
                val_losses.append(avg_val_loss)
                
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'Val Accuracy: {val_accuracy:.2f}%')
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Check for early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_epochs = 0
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': val_accuracy
                    }, 'models/best_spam_model.pt')
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"Early stopping after {epoch+1} epochs!")
                        break
                
                # Memory management
                torch.cuda.empty_cache()
                gc.collect()
            
            # Plot training history
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('visualizations/training_history.png')
            plt.close()
            
            # Final evaluation
            print("\nLoading best model for final evaluation...")
            checkpoint = torch.load('models/best_spam_model.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            test_correct = 0
            test_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in tqdm(test_loader, desc='Final Testing'):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                    
                    # Memory management
                    batch_x = batch_x.cpu()
                    batch_y = batch_y.cpu()
            
            test_accuracy = 100 * test_correct / test_total
            print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
            
            # Detailed classification metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            print("\nClassification Report:")
            report = classification_report(all_labels, all_predictions, target_names=['Ham', 'Spam'])
            print(report)
            
            # Save classification report
            with open('visualizations/classification_report.txt', 'w') as f:
                f.write(report)
            
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('visualizations/confusion_matrix.png')
            plt.close()
            
            # Save model metadata for inference
            metadata = {
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'num_classes': num_classes,
                'max_len': max_len,
                'accuracy': test_accuracy
            }
            
            # Save model artifacts for deployment
            print("\nSaving model artifacts for deployment...")
            with open('models/tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open('models/label_encoder.pickle', 'wb') as handle:
                pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open('models/metadata.pickle', 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Training completed successfully! All model artifacts saved.")
            
        except Exception as e:
            print(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in data processing: {e}")
        import traceback
        traceback.print_exc()