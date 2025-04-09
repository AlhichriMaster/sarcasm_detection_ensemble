import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential

class SarcasmModels:
    def __init__(self, vocab_size, max_len, embed_dim=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build_lstm_model(self):
        """Baseline LSTM model with regularization"""
        model = Sequential([
            layers.Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len),
            layers.Dropout(0.4), 
            layers.LSTM(256, return_sequences=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
        return model  
    
    def build_attention_model(self):
        """Attention-based model"""
        inputs = layers.Input(shape=(self.max_len,))
        embedding = layers.Embedding(self.vocab_size, self.embed_dim)(inputs)
        
        # Self-attention layer
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)(
            embedding, embedding)
        # attention = layers.LayerNormalization()(attention + embedding)
        
        lstm = layers.LSTM(64)(attention)
        dense = layers.Dense(32, activation='relu')(lstm)
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        return Model(inputs=inputs, outputs=output)
    
    def build_transformer_model(self, num_blocks=1):
        """Transformer-based model with configurable number of blocks"""
        inputs = layers.Input(shape=(self.max_len,))
        x = layers.Embedding(self.vocab_size, self.embed_dim)(inputs)
        
        # Stack multiple transformer blocks
        for _ in range(num_blocks):
            x = self._transformer_encoder(x)
        
        pooled = layers.GlobalAveragePooling1D()(x)
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
        ffn = layers.Dense(256, activation='relu')(attention)
        ffn = layers.Dense(self.embed_dim)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        
        return layers.LayerNormalization()(ffn + attention) 