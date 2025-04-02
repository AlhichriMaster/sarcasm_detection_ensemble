import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential

class SarcasmModels:
    def __init__(self, vocab_size, max_len, embed_dim=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build_lstm_model(self):
        """Baseline LSTM model"""
        model = Sequential([
            layers.Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_attention_model(self):
        """Attention-based model"""
        inputs = layers.Input(shape=(self.max_len,))
        embedding = layers.Embedding(self.vocab_size, self.embed_dim)(inputs)
        
        # Self-attention layer
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=self.embed_dim)(
            embedding, embedding)
        attention = layers.LayerNormalization()(attention + embedding)
        
        lstm = layers.LSTM(64)(attention)
        dense = layers.Dense(32, activation='relu')(lstm)
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        return Model(inputs=inputs, outputs=output)
    
    def build_transformer_model(self):
        """Transformer-based model"""
        inputs = layers.Input(shape=(self.max_len,))
        embedding = layers.Embedding(self.vocab_size, self.embed_dim)(inputs)
        
        # Transformer block
        transformer_block = self._transformer_encoder(embedding)
        
        pooled = layers.GlobalAveragePooling1D()(transformer_block)
        dense = layers.Dense(32, activation='relu')(pooled)
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        return Model(inputs=inputs, outputs=output)
    
    def _transformer_encoder(self, inputs):
        """Helper method to create transformer encoder block"""
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=self.embed_dim)(
            inputs, inputs)
        attention = layers.Dropout(0.1)(attention)
        attention = layers.LayerNormalization()(attention + inputs)
        
        # Feed-forward network
        ffn = layers.Dense(128, activation='relu')(attention)
        ffn = layers.Dense(self.embed_dim)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        
        return layers.LayerNormalization()(ffn + attention) 