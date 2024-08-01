import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from self_har_models import create_CNN_LSTM_Model
import dataset_pre_processing

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

def check_label_balance(labels, label_map):
    labels = labels.astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("Label balance:")
    for label, count in label_counts.items():
        label_name = next((name for name, idx in label_map.items() if idx == label), None)
        if label_name:
            print(f"{label}: {count} ({label_name})")
        else:
            print(f"Label {label} not found in label_map. Count: {count}")
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.show()

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

df_acc = dataset_pre_processing.concat_datasets(datasets, 'acc')
df_gyro = dataset_pre_processing.concat_datasets(datasets, 'gyro')

# Modify labels for sensor locations
labels_acc = dataset_pre_processing.get_labels(df_acc)
labels_gyro = dataset_pre_processing.get_labels(df_gyro)
labels = list(set(labels_acc) | set(labels_gyro))
label_map = {label: index for index, label in enumerate(labels)}
num_unique_labels = len(labels)
print(f"Number of unique labels: {num_unique_labels}")

callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Define inputs for each sensor type
acc_input = tf.keras.Input(shape=(400, 3), name='acc_input')
gyro_input = tf.keras.Input(shape=(400, 3), name='gyro_input')

# Create the CNN-LSTM model for feature extraction
acc_model = create_CNN_LSTM_Model((400, 3), model_name="CNN_LSTM_ACC")
gyro_model = create_CNN_LSTM_Model((400, 3), model_name="CNN_LSTM_GYRO")

# Extract features for each sensor type
acc_features = acc_model(acc_input)
gyro_features = gyro_model(gyro_input)

# Concatenate extracted features
combined_features = tf.keras.layers.Concatenate()([acc_features, gyro_features])

# Add dense layers and classification heads for sensor locations
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4))(combined_features)
x = tf.keras.layers.Dropout(0.5)(x)
acc_output = tf.keras.layers.Dense(num_unique_labels, activation='softmax', name='acc_output')(x)
gyro_output = tf.keras.layers.Dense(num_unique_labels, activation='softmax', name='gyro_output')(x)

# Create and compile the combined model
model = tf.keras.Model(inputs=[acc_input, gyro_input], outputs=[acc_output, gyro_output])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005),
              loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=['accuracy', 'accuracy'])

# Preprocess datasets for each sensor type, handling missing sensor types
def preprocess_sensor_data(sensor_type):
    available_datasets = [df for df in datasets if sensor_type in df]
    if not available_datasets:
        print(f"No available datasets for sensor type: {sensor_type}")
        return None
    sensor_df = dataset_pre_processing.concat_datasets(available_datasets, sensor_type)
    preprocessed_data = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=sensor_df,
        label_map=label_map,
        output_shape=len(label_map),
        train_users=users,
        test_users=[],
        window_size=400,
        shift=100,
        verbose=1
    )
    if not preprocessed_data:
        print(f"Preprocessing failed for sensor type: {sensor_type}")
    return preprocessed_data

user_dataset_preprocessed_acc = preprocess_sensor_data('acc')
user_dataset_preprocessed_gyro = preprocess_sensor_data('gyro')

if user_dataset_preprocessed_acc and user_dataset_preprocessed_gyro:
    acc_samples, acc_labels = user_dataset_preprocessed_acc[0]
    gyro_samples, gyro_labels = user_dataset_preprocessed_gyro[0]

    val_acc_samples, val_acc_labels = user_dataset_preprocessed_acc[1]
    val_gyro_samples, val_gyro_labels = user_dataset_preprocessed_gyro[1]


    # Check label balance
    print("Training Data:")
    check_label_balance(acc_labels, label_map)
    check_label_balance(gyro_labels, label_map)

    # Ensure the same number of samples for each sensor type
    min_samples = min(len(acc_samples), len(gyro_samples))
    acc_samples = acc_samples[:min_samples]
    gyro_samples = gyro_samples[:min_samples]
    acc_labels = acc_labels[:min_samples]
    gyro_labels = gyro_labels[:min_samples]

    min_val_samples = min(len(val_acc_samples), len(val_gyro_samples))
    val_acc_samples = val_acc_samples[:min_val_samples]
    val_gyro_samples = val_gyro_samples[:min_val_samples]
    val_acc_labels = val_acc_labels[:min_val_samples]
    val_gyro_labels = val_gyro_labels[:min_val_samples]

    # Train the model with both accelerometer and gyroscope labels
    history = model.fit(
        [acc_samples, gyro_samples],
        [acc_labels, gyro_labels],
        epochs=2,
        validation_data=(
            [val_acc_samples, val_gyro_samples],
            [val_acc_labels, val_gyro_labels]
        ),
        callbacks=[callback]
    )

    # Save the initial model
    model.save('pretext_combined_model.keras')

    # Predictions
    acc_pred_labels, gyro_pred_labels = model.predict([val_acc_samples, val_gyro_samples])
    acc_pred_labels = np.argmax(acc_pred_labels, axis=1)
    gyro_pred_labels = np.argmax(gyro_pred_labels, axis=1)
    val_acc_labels = np.argmax(val_acc_labels, axis=1)
    val_gyro_labels = np.argmax(val_gyro_labels, axis=1)

    # Plot confusion matrices
    plot_confusion_matrix(val_acc_labels, acc_pred_labels, classes=labels, title='Confusion Matrix for Accelerometer', filename='confusion_matrix_acc.png')
    plot_confusion_matrix(val_gyro_labels, gyro_pred_labels, classes=labels, title='Confusion Matrix for Gyroscope', filename='confusion_matrix_gyro.png')
else:
    print("One or more sensor types are missing in the datasets.")
