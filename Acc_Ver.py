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
df_acc = dataset_pre_processing.concat_datasets(datasets, 'acc')
df_gyro = dataset_pre_processing.concat_datasets(datasets, 'gyro')
users = list(df_acc.keys())

# Create separate label maps for accelerometer and gyroscope
labels_acc = dataset_pre_processing.get_labels(df_acc)
labels_gyro = dataset_pre_processing.get_labels(df_gyro)
label_map_acc = {label: index for index, label in enumerate(labels_acc)}
label_map_gyro = {label: index for index, label in enumerate(labels_gyro)}
num_unique_labels_acc = len(labels_acc)
num_unique_labels_gyro = len(labels_gyro)
print(f"Number of unique labels (ACC): {num_unique_labels_acc}")
print(f"Number of unique labels (GYRO): {num_unique_labels_gyro}")

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
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined_features)
x = tf.keras.layers.Dropout(0.5)(x)
acc_output = tf.keras.layers.Dense(num_unique_labels_acc, activation='softmax', name='acc_output')(x)
gyro_output = tf.keras.layers.Dense(num_unique_labels_gyro, activation='softmax', name='gyro_output')(x)

# Create and compile the combined model
model = tf.keras.Model(inputs=[acc_input, gyro_input], outputs=[acc_output, gyro_output])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=['accuracy', 'accuracy'])


# Preprocess datasets for each sensor type, handling missing sensor types
def preprocess_sensor_data(sensor_type, label_map):
    available_datasets = [df for df in datasets if sensor_type in df]
    if not available_datasets:
        print(f"No available datasets for sensor type: {sensor_type}")
        return None
    sensor_df = dataset_pre_processing.concat_datasets(available_datasets, sensor_type)
    print(f"Processing sensor data for type: {sensor_type}")

    # Check the initial labels
    initial_labels = dataset_pre_processing.get_labels(sensor_df)
    print(f"Initial labels for {sensor_type}: {initial_labels}")

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

    # Check the unique labels in the preprocessed data
    if preprocessed_data:
        train_data, val_data, test_data = preprocessed_data
        train_x, train_y = train_data
        val_x, val_y = val_data if val_data is not None else (None, None)
        test_x, test_y = test_data

        print(f"Unique labels in {sensor_type} train_y after preprocessing: {np.unique(train_y)}")
        if val_y is not None:
            print(f"Unique labels in {sensor_type} val_y after preprocessing: {np.unique(val_y)}")
        print(f"Unique labels in {sensor_type} test_y after preprocessing: {np.unique(test_y)}")

        # Check one-hot encoded labels
        one_hot_encoded_labels = tf.keras.utils.to_categorical(train_y, num_classes=len(label_map))
        print(f"Shape of one-hot encoded labels for {sensor_type}: {one_hot_encoded_labels.shape}")
        print(
            f"Unique labels in one-hot encoded data for {sensor_type}: {np.unique(np.argmax(one_hot_encoded_labels, axis=1))}")

    if not preprocessed_data:
        print(f"Preprocessing failed for sensor type: {sensor_type}")
    return preprocessed_data


# Preprocess the data for both accelerometer and gyroscope
acc_df = preprocess_sensor_data('acc', label_map_acc)
gyro_df = preprocess_sensor_data('gyro', label_map_gyro)


# Ensure the number of samples are the same for both accelerometer and gyroscope data
def balance_datasets(acc_data, gyro_data):
    acc_samples, acc_labels = acc_data
    gyro_samples, gyro_labels = gyro_data

    if len(acc_samples) > len(gyro_samples):
        # Oversample gyro data
        extra_samples = len(acc_samples) - len(gyro_samples)
        indices = np.random.choice(len(gyro_samples), extra_samples, replace=True)
        gyro_samples = np.concatenate([gyro_samples, gyro_samples[indices]], axis=0)
        gyro_labels = np.concatenate([gyro_labels, gyro_labels[indices]], axis=0)
    elif len(gyro_samples) > len(acc_samples):
        # Oversample acc data
        extra_samples = len(gyro_samples) - len(acc_samples)
        indices = np.random.choice(len(acc_samples), extra_samples, replace=True)
        acc_samples = np.concatenate([acc_samples, acc_samples[indices]], axis=0)
        acc_labels = np.concatenate([acc_labels, acc_labels[indices]], axis=0)

    return (acc_samples, acc_labels), (gyro_samples, gyro_labels)


(train_acc_x, train_acc_y), (train_gyro_x, train_gyro_y) = balance_datasets((acc_df[0][0], acc_df[0][1]),
                                                                            (gyro_df[0][0], gyro_df[0][1]))
(val_acc_x, val_acc_y), (val_gyro_x, val_gyro_y) = balance_datasets((acc_df[1][0], acc_df[1][1]),
                                                                    (gyro_df[1][0], gyro_df[1][1]))

# Print unique labels
print(f"Unique labels in train_acc_y: {np.unique(np.argmax(train_acc_y, axis=1))}")
print(f"Unique labels in val_acc_y: {np.unique(np.argmax(val_acc_y, axis=1))}")
print(f"Unique labels in train_gyro_y: {np.unique(np.argmax(train_gyro_y, axis=1))}")
print(f"Unique labels in val_gyro_y: {np.unique(np.argmax(val_gyro_y, axis=1))}")

# Train the model with both accelerometer and gyroscope labels
history = model.fit(
    [train_acc_x, train_gyro_x],
    [train_acc_y, train_gyro_y],
    epochs=100,
    validation_data=(
        [val_acc_x, val_gyro_x],
        [val_acc_y, val_gyro_y]
    ),
    callbacks=[callback]
)

# Save the initial model
model.save('pretext_combined_model.keras')

# Predictions
acc_pred_labels, gyro_pred_labels = model.predict([val_acc_x, val_gyro_x])
acc_pred_labels = np.argmax(acc_pred_labels, axis=1)
gyro_pred_labels = np.argmax(gyro_pred_labels, axis=1)
val_acc_labels = np.argmax(val_acc_y, axis=1)
val_gyro_labels = np.argmax(val_gyro_y, axis=1)

# Plot confusion matrices
plot_confusion_matrix(val_acc_labels, acc_pred_labels, classes=list(label_map_acc.keys()),
                      title='Confusion Matrix for Accelerometer', filename='confusion_matrix_acc.png')
plot_confusion_matrix(val_gyro_labels, gyro_pred_labels, classes=list(label_map_gyro.keys()),
                      title='Confusion Matrix for Gyroscope', filename='confusion_matrix_gyro.png')

print("One or more sensor types are missing in the datasets.")
