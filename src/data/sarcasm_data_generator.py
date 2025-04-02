import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tensorflow as tf

class SarcasmDataGenerator:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_tokenizer(self, texts):
        # Preprocess and fit tokenizer on training data
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.tokenizer.fit_on_texts(processed_texts)
    
    def generate_batches(self, texts, labels, batch_size=32, shuffle=True):
        """Generate data using tf.data.Dataset for better Keras compatibility"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
            
        dataset = dataset.batch(batch_size)
        
        # Only repeat for training data
        if shuffle:  # If shuffle is True, it's training data
            dataset = dataset.repeat()
        
        return dataset 