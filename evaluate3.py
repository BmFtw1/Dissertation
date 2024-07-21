import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import self_har_models
import pickle
import dataset_pre_processing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Evaluation1
import raw_data_processing

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
# with open('pickled_datasets/wisdm2.pickle', 'rb') as file:
#     wisdm_df = pickle.load(file)


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
# with open('pickled_datasets/wisdm_har.pickle', 'rb') as file:
#     wisdm_har_df = pickle.load(file)
with open('pickled_datasets/wisdm1_har.pickle', 'rb') as file:
    wisdm1_har_df = pickle.load(file)

df = dataset_pre_processing.concat_datasets([hhar_df, motion_sense_df, harth_df, dasa_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=21,
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
                                                                      output_shape=21,
                                                                      optimizer=tf.keras.optimizers.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([pamap_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
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
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 24,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("PAMAP")
print(all_info)

df = dataset_pre_processing.concat_datasets([hhar_df,  motion_sense_df, pamap_df, dasa_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=22,
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
                                                                      output_shape=22,
                                                                      optimizer=tf.keras.optimizers.legacy.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([harth_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
for i in range(3, user_train_size, 1):
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=har_df,
        label_map=har_label_map,
        output_shape=12,
        train_users=har_users[0:i],
        test_users=testing_users,
        window_size=400,
        shift=100
    )
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 12,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("HARTH")
print(all_info)

df = dataset_pre_processing.concat_datasets([hhar_df, motion_sense_df, pamap_df, harth_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=21,
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
                                                                      output_shape=21,
                                                                      optimizer=tf.keras.optimizers.legacy.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([dasa_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
for i in range(3, user_train_size, 1):
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=har_df,
        label_map=har_label_map,
        output_shape=19,
        train_users=har_users[0:i],
        test_users=testing_users,
        window_size=400,
        shift=10
    )
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 19,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("DASD")
print(all_info)

df = dataset_pre_processing.concat_datasets([pamap_df, motion_sense_df, harth_df, dasa_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=21,
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
                                                                      output_shape=21,
                                                                      optimizer=tf.keras.optimizers.legacy.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([hhar_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
for i in range(3, user_train_size, 1):
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=har_df,
        label_map=har_label_map,
        output_shape=6,
        train_users=har_users[0:i],
        test_users=testing_users,
        window_size=400,
        shift=100
    )
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 6,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("HHAR")
print(all_info)

df = dataset_pre_processing.concat_datasets([pamap_df, hhar_df, harth_df, dasa_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=33,
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
                                                                      output_shape=33,
                                                                      optimizer=tf.keras.optimizers.legacy.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([motionsense_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
for i in range(3, user_train_size, 2):
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=har_df,
        label_map=har_label_map,
        output_shape=6,
        train_users=har_users[0:i],
        test_users=testing_users,
        window_size=400,
        shift=100
    )
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 6,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("MOTIONSENSE")
print(all_info)

df = dataset_pre_processing.concat_datasets([pamap_df, hhar_df, motion_sense_df, harth_df, dasa_df], 'acc')
users = list(df.keys())
labels = dataset_pre_processing.get_labels(df)
label_map = {label: index for index, label in enumerate(labels)}
user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=df,
    label_map=label_map,
    output_shape=22,
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
                                                                      output_shape=22,
                                                                      optimizer=tf.keras.optimizers.legacy.Adam(
                                                                          learning_rate=0.0005))
history = composite_model.fit(df[0][0], df[0][1]
                              , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])

from Evaluation1 import downstream_testing, eval_model

har_df = dataset_pre_processing.concat_datasets([wisdm1_har_df], 'acc')
har_users = list(har_df.keys())

user_train_size = int(len(har_users) * .8)
training_users = har_users[0:(user_train_size)]

user_test_size = len(har_users) - user_train_size
testing_users = har_users[user_train_size:(user_train_size + user_test_size)]

labels = dataset_pre_processing.get_labels(har_df)
har_label_map = {label: index for index, label in enumerate(labels)}
all_info = []
for i in range(3, user_train_size, 2):
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=har_df,
        label_map=har_label_map,
        output_shape=6,
        train_users=har_users[0:i],
        test_users=testing_users,
        window_size=400,
        shift=100
    )
    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 6,
                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = eval_model(har_preprocessed, labels, har_model)
    print("Trained " + str(i) + " users")
    print(downstream_eval)
    info = "Trained " + str(i) + " users " + str(downstream_eval)
    all_info.append(info)
print("WISDM")
print(all_info)
