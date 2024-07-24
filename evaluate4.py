import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import self_har_models
import pickle
import dataset_pre_processing

import tensorflow as tf


tf.get_logger().setLevel('INFO')

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

df = dataset_pre_processing.concat_datasets([hhar_df, motion_sense_df, harth_df, dasa_df], 'all')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
num_unique_labels = len(labels)

user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=num_unique_labels,
    train_users=users,
    test_users=[],
    window_size=400,
    shift=100,
    verbose=1
)

cm = self_har_models.create_CNN_LSTM_Model((400, 3))
callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
df = user_dataset_preprocessed
composite_model = self_har_models.attach_full_har_classification_head(core_model=cm,
                                                                      output_shape=num_unique_labels,
                                                                      optimizer=tf.keras.optimizers.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])
# Save the initial model
composite_model.save('initial_model.h5')

