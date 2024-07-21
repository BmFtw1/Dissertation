import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import self_har_models
import dataset_pre_processing
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

tf.get_logger().setLevel('INFO')


def prepare_self_supervised_learning_dataset(sensor_type, train_users, test_users):
    with open('pickled_datasets/pamap2.pickle', 'rb') as file:
        pamap_df = pickle.load(file)
    with open('pickled_datasets/hhar2.pickle', 'rb') as file:
        hhar_df = pickle.load(file)
    with open('pickled_datasets/motionsense2.pickle', 'rb') as file:
        motion_sense_df = pickle.load(file)

    cdf = dataset_pre_processing.concat_datasets([pamap_df, hhar_df, motion_sense_df], sensor_type=sensor_type)
    labels = dataset_pre_processing.get_labels(cdf)
    label_map = {label: index for index, label in enumerate(labels)}
    user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=cdf,
        label_map=label_map,
        output_shape=14,
        train_users=train_users,
        test_users=test_users,
        window_size=400,
        shift=200
    )
    print(labels)
    return user_dataset_preprocessed, labels


def train_self_supervised_model(df, core_model, label_size, optimizer):
    print("Pseudo-labels for self-supervised training:")
    print(np.unique(df[0][1]))  # Assuming df[0][1] contains the labels
    callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    composite_model = self_har_models.attach_full_har_classification_head(core_model=core_model,
                                                                          output_shape=label_size,
                                                                          optimizer=optimizer)
    history = composite_model.fit(df[0][0], df[0][1]
                                  , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])
    return history, composite_model


def eval_model(df, labels, model):
    # cnn_test_result = model.evaluate(df[2][0],  df[2][1], return_dict=True)
    predicted_labels = np.argmax(model.predict(df[2][0]), axis=1)
    true_labels = np.argmax(df[2][1], axis=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f1)
    confusion_mat = tf.math.confusion_matrix(true_labels, predicted_labels)
    cm = confusion_mat.numpy()
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.plot()
    plt.show()
    return f1


def downstream_testing(df, model, label_size, optimizer):
    callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    core_model = self_har_models.extract_core_model(model)
    cm = tf.keras.models.clone_model(core_model)
    har_model = self_har_models.attach_full_har_classification_head(core_model=cm,
                                                                    output_shape=label_size,
                                                                    optimizer=optimizer)
    history = har_model.fit(df[0][0], df[0][1]
                            , epochs=100, validation_data=(df[1][0], df[1][1]), callbacks=[callback])
    return history, har_model


def eval_downstream_model(df, har_df, sensor_type, har_sensor_type, training_users=None, testing_users=None,
                          core_model='CNN', step=1, shift=100):
    df = dataset_pre_processing.concat_datasets([df], sensor_type=sensor_type)
    outputshape = len(set(df[list(df.keys())[0]][0][1]))
    users = list(df.keys())

    if training_users == None:
        user_train_size = int(len(users) * .8)
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
        cm = self_har_models.create_CNN_LSTM_Model((400, 3))
    elif core_model == 'LSTM_CNN':
        cm = self_har_models.create_LSTM_CNN_Model((400, 3))
    elif core_model == 'LSTM':
        cm = self_har_models.create_LSTM_Model((400, 3))
    elif core_model == 'Transformer':
        cm = self_har_models.create_transformer_model((400, 3))
    elif core_model == 'CNN':
        cm = self_har_models.create_1d_conv_core_model((400, 3))
    else:
        print('cannot find model, training CNN-LSTM model isntead')
        cm = self_har_models.create_CNN_LSTM_Model((400, 3))

    history, composite_model = train_self_supervised_model(user_dataset_preprocessed, cm, outputshape,
                                                           tf.keras.optimizers.Adam(learning_rate=0.0005))

    eval_model(user_dataset_preprocessed, labels, composite_model)

    har_df = dataset_pre_processing.concat_datasets([har_df], sensor_type=har_sensor_type)
    outputshape2 = len(set(har_df[list(har_df.keys())[0]][0][1]))
    har_labels = dataset_pre_processing.get_labels(har_df)
    outputshape2 = len(har_labels)
    har_label_map = {label: index for index, label in enumerate(har_labels)}
    all_info = []
    for i in range(3, user_train_size, step):
        har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
            user_datasets=har_df,
            label_map=har_label_map,
            output_shape=outputshape2,
            train_users=users[0:i],
            test_users=testing_users,
            window_size=400,
            shift=shift,
            verbose=1
        )
        ds_history, har_model = downstream_testing(har_preprocessed, composite_model, outputshape2,
                                                   tf.keras.optimizers.Adam(learning_rate=0.0005))
        downstream_eval = eval_model(har_preprocessed, har_labels, har_model)
        print("Trained " + str(i) + " users")
        print(downstream_eval)
        info = "Trained " + str(i) + " users " + str(downstream_eval)
        all_info.append(info)
    print("\n")
    return (all_info)


def eval_multi_model(df_list, har_df_list, output_shape, har_output_shape, sensor_type: "all", training_users=None,
                     testing_users=None, core_model='CNN_LSTM', step=2):
    df = dataset_pre_processing.concat_datasets(df_list, sensor_type=sensor_type)
    outputshape = output_shape
    users = list(df.keys())

    if training_users == None:
        user_train_size = int(len(users) * .8)
        training_users = users[0:(user_train_size)]
    else:
        user_train_size = len(training_users)

    if testing_users == None:
        user_test_size = len(users) - user_train_size
        testing_users = users[user_train_size:(user_train_size + user_test_size)]
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
        shift=200
    )
    if core_model == 'CNN_LSTM':
        cm = self_har_models.create_CNN_LSTM_Model((400, 3))
    elif core_model == 'LSTM_CNN':
        cm = self_har_models.create_LSTM_CNN_Model((400, 3))
    elif core_model == 'LSTM':
        cm = self_har_models.create_LSTM_Model((400, 3))
    elif core_model == 'Transformer':
        cm = self_har_models.create_transformer_model((400, 3))
    else:
        print('cannot find model, training CNN-LSTM model isntead')
        cm = self_har_models.create_CNN_LSTM_Model((400, 3))

    history, composite_model = train_self_supervised_model(user_dataset_preprocessed, cm, outputshape,
                                                           tf.keras.optimizers.Adam(learning_rate=0.0005))

    eval_model(user_dataset_preprocessed, labels, composite_model)

    har_df = dataset_pre_processing.concat_datasets(har_df_list, sensor_type=sensor_type)
    outputshape2 = har_output_shape
    har_labels = dataset_pre_processing.get_labels(har_df)

    har_label_map = {label: index for index, label in enumerate(har_labels)}
    all_info = []
    for i in range(3, user_train_size, step):
        har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
            user_datasets=har_df,
            label_map=har_label_map,
            output_shape=outputshape2,
            train_users=users[0:i],
            test_users=testing_users,
            window_size=400,
            shift=200
        )
        ds_history, har_model = downstream_testing(har_preprocessed, composite_model, outputshape2,
                                                   tf.keras.optimizers.Adam(learning_rate=0.0005))
        downstream_eval = eval_model(har_preprocessed, har_labels, har_model)
        print("Trained " + str(i) + " users")
        print(downstream_eval)
        info = "Trained " + str(i) + " users " + str(downstream_eval)
        all_info.append(info)
    print("\n")
    return (all_info)


def eval_fully_supervised(df, shift=100, step=1):
    df = dataset_pre_processing.concat_datasets([df], sensor_type="acc")
    users = list(df.keys())

    user_train_size = int(len(users) * .8)
    training_users = users[0:(user_train_size)]
    print(training_users)

    user_test_size = len(users) - user_train_size
    testing_users = users[user_train_size:(user_train_size + user_test_size)]
    print(testing_users)

    labels = dataset_pre_processing.get_labels(df)
    outputshape = len(labels)
    label_map = {label: index for index, label in enumerate(labels)}
    all_info = []
    for i in range(3, user_train_size, step):
        user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
            user_datasets=df,
            label_map=label_map,
            output_shape=outputshape,
            train_users=users[0:i],
            test_users=testing_users,
            window_size=400,
            shift=shift,
            verbose=1
        )
        cm = self_har_models.create_CNN_LSTM_Model((400, 3))

        history, composite_model = train_self_supervised_model(user_dataset_preprocessed, cm, outputshape,
                                                               tf.keras.optimizers.Adam(learning_rate=0.0005))
        eval = eval_model(user_dataset_preprocessed, labels, composite_model)
        info = "Trained " + str(i) + " users " + str(eval)
        all_info.append(info)
    return all_info


def eval_harth():
    with open('pickled_datasets/pamap2.pickle', 'rb') as file:
        pamap_df = pickle.load(file)
    with open('pickled_datasets/hhar2.pickle', 'rb') as file:
        hhar_df = pickle.load(file)
    with open('pickled_datasets/motionsense2.pickle', 'rb') as file:
        motion_sense_df = pickle.load(file)

    cdf = dataset_pre_processing.concat_datasets([pamap_df, hhar_df, motion_sense_df], "acc")
    labels = dataset_pre_processing.get_labels(cdf)
    label_map = {label: index for index, label in enumerate(labels)}
    user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        cdf, label_map, 6,
        ['101', '102', '103', '104', '105', '106', '107', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '1', '2', '3', '4', '5',
         '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', "108", "109", "h", "i",
         "21", "22", "23", "24"],
        [], 400, 200
    )
    callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    cm = self_har_models.create_CNN_LSTM_Model((400, 3))
    history, composite_model = train_self_supervised_model(user_dataset_preprocessed, cm, 14,
                                                           tf.keras.optimizers.Adam(learning_rate=0.0005))
