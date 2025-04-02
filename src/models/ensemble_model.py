import tensorflow as tf
from tensorflow.keras import layers, Model

class SarcasmEnsemble:
    def __init__(self, models):
        self.models = models
    
    def build_ensemble(self):
        """Create an ensemble model that averages predictions"""
        inputs = layers.Input(shape=(self.models[0].input_shape[1],))
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            predictions.append(model(inputs))
        
        # Average predictions
        ensemble_output = layers.Average()(predictions)
        
        return Model(inputs=inputs, outputs=ensemble_output) 