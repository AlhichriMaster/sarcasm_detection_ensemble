import json
from data.sarcasm_data_generator import SarcasmDataGenerator
from models.ensemble_model import SarcasmEnsemble
from models.sarcasm_models import SarcasmModels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, dataset, steps, model_name):
    """Evaluate model and return predictions and metrics"""
    # Get predictions
    predictions = model.predict(dataset, steps=steps)
    y_pred = (predictions > 0.5).astype(int)
    
    # Get true labels from the original test data
    y_true = test_labels[:len(y_pred)]
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name, 
                         os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))
    
    return y_pred, report

# Load the JSON dataset
def load_sarcasm_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            texts.append(item['headline'])
            labels.append(1 if item['is_sarcastic'] else 0)
    return np.array(texts), np.array(labels)

# Load and prepare the data
texts, labels = load_sarcasm_dataset('./Sarcasm_Headlines_Dataset.json')

# Split data into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Initialize data generator
data_generator = SarcasmDataGenerator()
data_generator.fit_tokenizer(train_texts)

# Create data generators for training
BATCH_SIZE = 32
train_dataset = data_generator.generate_batches(train_texts, train_labels, batch_size=BATCH_SIZE)
val_dataset = data_generator.generate_batches(val_texts, val_labels, batch_size=BATCH_SIZE)

# Initialize models
model_builder = SarcasmModels(
    vocab_size=data_generator.max_words,
    max_len=data_generator.max_len
)

# Build individual models
models = {
    'LSTM': model_builder.build_lstm_model(),
    'Attention': model_builder.build_attention_model(),
    'Transformer': model_builder.build_transformer_model()
}

# Calculate steps per epoch
steps_per_epoch = len(train_texts) // BATCH_SIZE
validation_steps = len(val_texts) // BATCH_SIZE

# Train and evaluate individual models
model_histories = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    model_histories[model_name] = history.history
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_history_{model_name}.png'))
    plt.close()

# Create and train ensemble
print("\nTraining ensemble model...")
ensemble = SarcasmEnsemble(list(models.values()))
ensemble_model = ensemble.build_ensemble()
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare test data
test_dataset = data_generator.generate_batches(test_texts, test_labels, batch_size=BATCH_SIZE, shuffle=False)
test_steps = len(test_texts) // BATCH_SIZE

# Evaluate all models
print("\nEvaluating models...")
results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name} model...")
    y_pred, report = evaluate_model(model, test_dataset, test_steps, model_name)
    results[model_name] = report
    print(f"\n{model_name} Model Results:")
    print(report)

# Evaluate ensemble
print("\nEvaluating ensemble model...")
ensemble_pred, ensemble_report = evaluate_model(ensemble_model, test_dataset, test_steps, 'Ensemble')
results['Ensemble'] = ensemble_report
print("\nEnsemble Model Results:")
print(ensemble_report)

# Save results to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(os.path.join(results_dir, f'model_results_{timestamp}.txt'), 'w') as f:
    f.write("Model Evaluation Results\n")
    f.write("======================\n\n")
    for model_name, report in results.items():
        f.write(f"\n{model_name} Model Results:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
        f.write("\n")

print(f"\nResults have been saved to the '{results_dir}' directory.")