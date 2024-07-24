import pickle
import tensorflow as tf

import dataset_pre_processing
import self_har_models

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

with open('pickled_datasets/pamap_har.pickle', 'rb') as file:
    pamap_har_df = pickle.load(file)
with open('pickled_datasets/hhar_har.pickle', 'rb') as file:
    hhar_har_df = pickle.load(file)
with open('pickled_datasets/motionsense_har.pickle', 'rb') as file:
    motionsense_har_df = pickle.load(file)
with open('pickled_datasets/harth_har.pickle', 'rb') as file:
    harth_har_df = pickle.load(file)
with open('pickled_datasets/dasa_har.pickle', 'rb') as file:
    dasa_har_df = pickle.load(file)
with open('pickled_datasets/wisdm1_har.pickle', 'rb') as file:
    wisdm1_har_df = pickle.load(file)

# Check if all sensor types are available in each dataset
datasets = [hhar_df, motion_sense_df, harth_df, dasa_df]
for i, df in enumerate(datasets):
    print(f"Dataset {i+1} keys: {df.keys()}")

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
mag_input = tf.keras.Input(shape=(400, 3), name='mag_input')

# Create the shared model
shared_model = self_har_models.create_CNN_LSTM_Model((400, 3))

# Extract features for each sensor type
acc_features = shared_model(acc_input)
gyro_features = shared_model(gyro_input)
mag_features = shared_model(mag_input)

# Concatenate extracted features
combined_features = tf.keras.layers.Concatenate()([acc_features, gyro_features, mag_features])

# Add dense layers and classification head
x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
final_output = tf.keras.layers.Dense(num_unique_labels, activation='softmax')(x)

# Create and compile the model
model = tf.keras.Model(inputs=[acc_input, gyro_input, mag_input], outputs=final_output)
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
user_dataset_preprocessed_mag = preprocess_sensor_data('mag')

# Ensure all datasets are available for training
if user_dataset_preprocessed_acc and user_dataset_preprocessed_gyro and user_dataset_preprocessed_mag:
    # Train the model
    history = model.fit(
        [user_dataset_preprocessed_acc[0][0], user_dataset_preprocessed_gyro[0][0], user_dataset_preprocessed_mag[0][0]],
        user_dataset_preprocessed_acc[0][1],
        epochs=100,
        validation_data=(
            [user_dataset_preprocessed_acc[1][0], user_dataset_preprocessed_gyro[1][0], user_dataset_preprocessed_mag[1][0]],
            user_dataset_preprocessed_acc[1][1]
        ),
        callbacks=[callback]
    )

    # Save the initial model
    model.save('initial_model.h5')
else:
    print("One or more sensor types are missing in the datasets.")
