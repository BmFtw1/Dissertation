import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from self_har_models import create_CNN_LSTM_Model, attach_full_har_classification_head
import dataset_pre_processing

# Centralize loading datasets
def load_datasets(paths):
    datasets = []
    for path in paths:
        with open(path, 'rb') as file:
            datasets.append(pickle.load(file))
    return datasets

dataset_paths = [
    'pickled_datasets/hhar2.pickle',
    'pickled_datasets/motionsense2.pickle',
    'pickled_datasets/harth2.pickle',
    'pickled_datasets/dasa2.pickle',
    'pickled_datasets/pamap2.pickle'
]

print("Loading datasets...")
datasets = load_datasets(dataset_paths)

# Concatenate accelerometer datasets
df_acc = dataset_pre_processing.concat_datasets(datasets, 'acc')
users = list(df_acc.keys())

# Get labels and create label map
labels = dataset_pre_processing.get_labels(df_acc)
label_map = {label: index for index, label in enumerate(labels)}
print(f"Label map: {label_map}")

# Preprocess dataset
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df_acc,
    label_map=label_map,
    output_shape=len(label_map),  # Use the correct number of unique labels
    train_users=users,
    test_users=[],  # Intentionally empty
    window_size=400,
    shift=100,
    verbose=1
)

# Debugging output
train_data, val_data, _ = user_dataset_preprocessed
train_x, train_y = train_data
val_x, val_y = val_data

print(f"Training data shape: {train_x.shape}, {train_y.shape}")
print(f"Validation data shape: {val_x.shape}, {val_y.shape}")
print(f"Unique labels in train_y: {np.unique(train_y)}")
print(f"Unique labels in val_y: {np.unique(val_y)}")

# Create and compile the CNN-LSTM model
cm = create_CNN_LSTM_Model((400, 3))
callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Attach full HAR classification head
composite_model = attach_full_har_classification_head(
    core_model=cm,
    output_shape=len(label_map),  # Use the correct number of unique labels
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003)
)

# Train the model
history = composite_model.fit(
    train_x, train_y,
    epochs=100,
    validation_data=(val_x, val_y),
    callbacks=[callback]
)

# Check the training history
print(f"Training history: {history.history}")
