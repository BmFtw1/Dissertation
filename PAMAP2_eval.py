import pickle

import dataset_pre_processing
import self_har_models
from Evaluation1 import downstream_testing, eval_model
import tensorflow as tf

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
# Function to replace the classification head
def replace_classification_head(model, output_shape, optimizer):
    core_model = model.layers[0]
    new_model = self_har_models.attach_full_har_classification_head(core_model=core_model,
                                                                    output_shape=output_shape,
                                                                    optimizer=optimizer)
    return new_model

# Load the saved initial model
composite_model = tf.keras.models.load_model('initial_model.h5', custom_objects={'tf': tf})

# Downstream task evaluation for PAMAP
har_df = dataset_pre_processing.concat_datasets([pamap_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]
user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []

# Open the file in write mode
with open('output.txt', 'w') as file:
    for i in range(3, user_train_size, 1):
        har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
            user_datasets=har_df,
            label_map=har_label_map,
            output_shape=24,
            train_users=har_users[0:i],
            test_users=testing_users,
            window_size=400,
            shift=100
        )
        composite_model = replace_classification_head(composite_model, 24, tf.keras.optimizers.Adam(learning_rate=0.0005))
        ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 24,
                                                   tf.keras.optimizers.Adam(learning_rate=0.0005))
        downstream_eval = eval_model(har_preprocessed, labels, har_model)
        info = f"Trained {i} users {downstream_eval}\n"

        # Write info to file
        file.write(info)
        all_info.append(info)

    # Write summary to file
    file.write("PAMAP\n")
    for info in all_info:
        file.write(info)

print("Output written to output.txt")
