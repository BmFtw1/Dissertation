import os
from Evaluation1 import downstream_testing, eval_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import self_har_models
import pickle
import dataset_pre_processing
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# Load datasets
print("Loading datasets...")
with open('pickled_datasets/pamap2.pickle', 'rb') as file:
    pamap_df = pickle.load(file)
with open('pickled_datasets/hhar2.pickle', 'rb') as file:
    hhar_df = pickle.load(file)
with open('pickled_datasets/motionsense2.pickle', 'rb') as file:
    motion_sense_df = pickle.load(file)
with open('pickled_datasets/harth2.pickle', 'rb') as file:
    harth_df = pickle.load(file)
with open('pickled_datasets/dasa2.pickle', 'rb') as file:
    dasa_df = pickle.load(file)

# Check if all sensor types are available in each dataset
datasets = [hhar_df, motion_sense_df, harth_df, dasa_df, pamap_df]
for i, df in enumerate(datasets):
    print(f"Dataset {i + 1} keys: {df.keys()}")

# Concatenate datasets
print("Concatenating datasets...")
df = dataset_pre_processing.concat_datasets(datasets, 'all')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
num_unique_labels = len(labels)
print(f"Number of unique labels: {num_unique_labels}")

callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Define inputs for each sensor type
acc_input = tf.keras.Input(shape=(400, 3), name='acc_input')
gyro_input = tf.keras.Input(shape=(400, 3), name='gyro_input')

# Create the shared model
shared_model = self_har_models.create_CNN_LSTM_Model((400, 3))

# Extract features for each sensor type
acc_features = shared_model(acc_input)
gyro_features = shared_model(gyro_input)

# Concatenate extracted features
combined_features = tf.keras.layers.Concatenate()([acc_features, gyro_features])

# Add dense layers and classification head
x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
final_output = tf.keras.layers.Dense(num_unique_labels, activation='softmax')(x)

# Create and compile the model
model = tf.keras.Model(inputs=[acc_input, gyro_input], outputs=final_output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Preprocess datasets for each sensor type, handling missing sensor types
def preprocess_sensor_data(sensor_type):
    available_datasets = [df for df in datasets if sensor_type in df]
    if not available_datasets:
        return None
    sensor_df = dataset_pre_processing.concat_datasets(available_datasets, sensor_type)
    return dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=sensor_df,
        label_map=label_map,
        output_shape=num_unique_labels,
        train_users=users,
        test_users=[],
        window_size=400,
        shift=100,
        verbose=1
    )


user_dataset_preprocessed_acc = preprocess_sensor_data('acc')
user_dataset_preprocessed_gyro = preprocess_sensor_data('gyro')

# Ensure all datasets are available for training
if user_dataset_preprocessed_acc and user_dataset_preprocessed_gyro:
    acc_samples, acc_labels = user_dataset_preprocessed_acc[0]
    gyro_samples, gyro_labels = user_dataset_preprocessed_gyro[0]

    # Ensure the same number of samples for each sensor type
    min_samples = min(len(acc_samples), len(gyro_samples))
    acc_samples, gyro_samples = acc_samples[:min_samples], gyro_samples[:min_samples]
    acc_labels, gyro_labels = acc_labels[:min_samples], gyro_labels[:min_samples]

    val_acc_samples, val_acc_labels = user_dataset_preprocessed_acc[1]
    val_gyro_samples, val_gyro_labels = user_dataset_preprocessed_gyro[1]

    min_val_samples = min(len(val_acc_samples), len(val_gyro_samples))
    val_acc_samples, val_gyro_samples = val_acc_samples[:min_val_samples], val_gyro_samples[:min_val_samples]
    val_acc_labels, val_gyro_labels = val_acc_labels[:min_val_samples], val_gyro_labels[:min_val_samples]

    # Print training and validation data distribution
    def print_label_distribution(labels, label_names):
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        for label, count in distribution.items():
            print(f"{label_names[label]}: {count}")

    print("Training label distribution:")
    train_labels = np.argmax(acc_labels, axis=1)  # Convert one-hot to single labels
    print_label_distribution(train_labels, labels)
    print(f"acc_labels are {acc_labels}")
    print("Validation label distribution:")
    val_labels = np.argmax(val_acc_labels, axis=1)  # Convert one-hot to single labels
    print_label_distribution(val_labels, labels)
    print(f"val_labels are {val_acc_labels}")
    # Train the model
    history = model.fit(
        [acc_samples, gyro_samples],
        acc_labels,
        epochs=1,
        validation_data=(
            [val_acc_samples, val_gyro_samples],
            val_acc_labels
        ),
        callbacks=[callback]
    )

    # Save the initial model
    model.save('combined_model.keras')
else:
    print("One or more sensor types are missing in the datasets.")
