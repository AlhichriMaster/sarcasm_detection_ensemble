import tensorflow as tf
from data.sarcasm_data_generator import SarcasmDataGenerator
from models.ensemble_model import SarcasmEnsemble
from models.sarcasm_models import SarcasmModels
import numpy as np
import json

def load_models():
    # Load the dataset to initialize the tokenizer
    texts = []
    with open('./Sarcasm_Headlines_Dataset.json', 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            texts.append(item['headline'])
    
    # Initialize data generator and fit tokenizer
    data_generator = SarcasmDataGenerator()
    data_generator.fit_tokenizer(texts)
    
    # Initialize models
    model_builder = SarcasmModels(
        vocab_size=data_generator.max_words,
        max_len=data_generator.max_len
    )
    
    # Build models
    models = {
        'LSTM': model_builder.build_lstm_model(),
        'Attention': model_builder.build_attention_model(),
        'Transformer': model_builder.build_transformer_model(1)
    }
    
    # Build each model with a dummy input to initialize weights
    dummy_input = tf.keras.Input(shape=(data_generator.max_len,))
    for model in models.values():
        model(dummy_input)
    
    # Load weights (assuming models are saved in a 'saved_models' directory)
    for model_name, model in models.items():
        model.load_weights(f'saved_models/{model_name.lower()}.weights.h5')
    
    # Create ensemble
    ensemble = SarcasmEnsemble(list(models.values()))
    ensemble_model = ensemble.build_ensemble()
    
    # Build ensemble model
    ensemble_model(dummy_input)
    
    return models, ensemble_model, data_generator

def predict_sarcasm(text, models, ensemble_model, data_generator):
    # Preprocess text
    processed_text = data_generator.preprocess_text(text)
    sequence = data_generator.tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=data_generator.max_len
    )
    
    # Get predictions from all models
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(padded_sequence)[0][0]
        predictions[model_name] = {
            'probability': float(pred),
            'is_sarcastic': pred > 0.5
        }
    
    # Get ensemble prediction
    ensemble_pred = ensemble_model.predict(padded_sequence)[0][0]
    predictions['Ensemble'] = {
        'probability': float(ensemble_pred),
        'is_sarcastic': ensemble_pred > 0.5
    }
    
    return predictions

def main():
    print("Loading models...")
    models, ensemble_model, data_generator = load_models()
    print("Models loaded successfully!")
    
    print("\nWelcome to the Sarcasm Detection Demo!")
    print("Enter a headline to check if it's sarcastic (or 'quit' to exit):")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
            
        predictions = predict_sarcasm(text, models, ensemble_model, data_generator)
        
        print("\nPrediction Results:")
        print("-" * 50)
        for model_name, pred in predictions.items():
            result = "Sarcastic" if pred['is_sarcastic'] else "Not Sarcastic"
            print(f"{model_name}: {result} (confidence: {pred['probability']:.2%})")
        print("-" * 50)

if __name__ == "__main__":
    main() 