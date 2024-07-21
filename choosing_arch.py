"""
This python script is used to choose which architecture to use.
"""

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
tf.get_logger().setLevel('INFO')

def eval_downstream_model(df, har_df, sensor_type, har_sensor_type, training_users=None, testing_users=None, core_model='CNN_LSTM', step=1, shift=100):
    df = dataset_pre_processing.concat_datasets([df], sensor_type=sensor_type)
    outputshape = len(set(df[list(df.keys())[0]][0][1]))
    users = list(df.keys())
    
    if training_users == None:
        user_train_size = int(len(users)*.8)
        training_users = users[0:(user_train_size)]
        print(training_users)
    else:
        user_train_size = len(training_users)

    if testing_users == None:
        user_test_size = len(users) - user_train_size
        testing_users = users[user_train_size:(user_train_size + user_test_size)]
        print(testing_users)
    else:
        user_test_size = len(testing_users)
    labels = dataset_pre_processing.get_labels(df)
    label_map = {label: index for index, label in enumerate(labels)}
    user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=df, 
        label_map=label_map, 
        output_shape=outputshape,
        train_users=training_users,
        test_users=testing_users,
        window_size=400, 
        shift=shift, 
        verbose=1
    )
    if core_model == 'CNN_LSTM':
        cm = self_har_models.create_CNN_LSTM_Model((400,3))
    elif core_model == 'LSTM_CNN':
        cm = self_har_models.create_LSTM_CNN_Model((400,3))
    elif core_model == 'LSTM':
        cm = self_har_models.create_LSTM_Model((400,3))
    elif core_model == 'Transformer':
        cm = self_har_models.create_transformer_model((400,3))
    elif core_model == 'CNN':
        cm = self_har_models.create_1d_conv_core_model((400,3))
    else:
        print('cannot find model, training CNN-LSTM model isntead')
        cm = self_har_models.create_CNN_LSTM_Model((400,3))

    history, composite_model = Evaluation1.train_self_supervised_model(user_dataset_preprocessed, cm, outputshape, 
                                                           tf.keras.optimizers.Adam(learning_rate=0.0005))
    
    Evaluation1.eval_model(user_dataset_preprocessed, labels, composite_model)

    har_df = dataset_pre_processing.concat_datasets([har_df], sensor_type=har_sensor_type)
    outputshape2 = len(set(har_df[list(har_df.keys())[0]][0][1]))
    har_labels = dataset_pre_processing.get_labels(har_df)
    outputshape2 = len(har_labels)
    har_label_map = {label: index for index, label in enumerate(har_labels)}
    all_info = []
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
    user_datasets=har_df,
    label_map=har_label_map,
    output_shape=outputshape2,
    train_users=training_users,
    test_users=testing_users,
    window_size=400,
    shift=shift,
    verbose=1
    )
    ds_history, har_model = Evaluation1.downstream_testing(har_preprocessed, composite_model, outputshape2, 
                                           tf.keras.optimizers.Adam(learning_rate=0.0005))
    downstream_eval = Evaluation1.eval_model(har_preprocessed, har_labels, har_model)
    return (downstream_eval)


with open('pickled_datasets/pamap2.pickle', 'rb') as file:
    hhar_df = pickle.load(file)
with open('pickled_datasets/pamap_har.pickle', 'rb') as file:
    hhar_har_df = pickle.load(file)
    

column_names = ['LSTM', 'LSTM_CNN', 'CNN', 'CNN_LSTM', 'Transformer']  

results = []

for _ in range(3):
    for core_model in column_names:
        result = eval_downstream_model(hhar_df, hhar_har_df, 'acc', 'acc', core_model=core_model)
        results.append((core_model, result))

print(result)