import glob
import re
import os
import pandas as pd
import numpy as np
import pickle

"""
This code was adapted from SelfHAR for the project's uses. 
"""

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{10.1145/3448112,
  author = {Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  title = {SelfHAR: Improving Human Activity Recognition through Self-Training with Unlabeled Data},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {5},
  number = {1},
  url = {https://doi.org/10.1145/3448112},
  doi = {10.1145/3448112},
  abstract = {Machine learning and deep learning have shown great promise in mobile sensing applications, including Human Activity Recognition. However, the performance of such models in real-world settings largely depends on the availability of large datasets that captures diverse behaviors. Recently, studies in computer vision and natural language processing have shown that leveraging massive amounts of unlabeled data enables performance on par with state-of-the-art supervised models.In this work, we present SelfHAR, a semi-supervised model that effectively learns to leverage unlabeled mobile sensing datasets to complement small labeled datasets. Our approach combines teacher-student self-training, which distills the knowledge of unlabeled and labeled datasets while allowing for data augmentation, and multi-task self-supervision, which learns robust signal-level representations by predicting distorted versions of the input.We evaluated SelfHAR on various HAR datasets and showed state-of-the-art performance over supervised and previous semi-supervised approaches, with up to 12% increase in F1 score using the same number of model parameters at inference. Furthermore, SelfHAR is data-efficient, reaching similar performance using up to 10 times less labeled data compared to supervised approaches. Our work not only achieves state-of-the-art performance in a diverse set of HAR datasets, but also sheds light on how pre-training tasks may affect downstream performance.},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  month = mar,
  articleno = {36},
  numpages = {30},
  keywords = {semi-supervised training, human activity recognition, unlabeled data, self-supervised training, self-training, deep learning}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk

Copyright (C) 2021 C. I. Tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)

        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                # user_trial_dataset.dropna(how="any", inplace=True)

                # Extract the x, y, z channels
                values = user_trial_dataset[['x', 'y', 'z']].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    print(user_datasets)
    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    
    """
    # print(data_folder_path)

    har_dataset = pd.read_csv(
        os.path.join(data_folder_path, 'Phones_accelerometer.csv'))  # "<PATH_TO_HHAR_DATASET>/Phones_accelerometer.csv"
    har_dataset.dropna(how="any", inplace=True)
    har_dataset = har_dataset[['x-axis', 'y-axis', 'z-axis', "gt", "User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data, labels)]

    return user_datasets


def process_hhar_all_files(data_folder_path):
    """
    Preprocess all files from the HHAR dataset.
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    Args:
        data_folder_path: folder path

    Returns:
        user_datasets (a list containing data collected from the acc + gyro from both smartphones and watches)
    """
    # Mapping devices to body parts
    device_to_body_part = {
        'nexus4_1': 'hand',
        'nexus4_2': 'hand',
        'lgwatch_1': 'wrist',
        'lgwatch_2': 'wrist',
        'gear_1': 'wrist',
        'gear_2': 'wrist',
        's3_1': 'hand',
        's3_2': 'hand',
        's3mini_1': 'hand',
        's3mini_2': 'hand',
        'samsungold_1': 'hand',
        'samsungold_2': 'hand'
    }

    # Loading data
    har_phone_acc = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    har_phone_gyro = pd.read_csv(os.path.join(data_folder_path, 'Phones_gyroscope.csv'))
    har_watch_acc = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
    har_watch_gyro = pd.read_csv(os.path.join(data_folder_path, 'Watch_gyroscope.csv'))

    # Combining data
    acc_data = pd.concat([har_phone_acc, har_watch_acc])
    gyro_data = pd.concat([har_phone_gyro, har_watch_gyro])

    # Dropping missing values and renaming columns
    acc_data.dropna(how="any", inplace=True)
    acc_data = acc_data[['x', 'y', 'z', "gt", "User", "Device"]]
    acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    har_users = acc_data["user-id"].unique()

    # Creating pseudo-labels for accelerometer data
    acc_data['body_part'] = acc_data['device'].map(device_to_body_part)
    acc_data['pseudo_label'] = acc_data['body_part'] + '_acc'

    acc_datasets = {}
    for user in har_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        labels = user_extract["pseudo_label"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    # Dropping missing values and renaming columns for gyroscope data
    gyro_data.dropna(how="any", inplace=True)
    gyro_data = gyro_data[['x', 'y', 'z', "gt", "User", "Device"]]
    gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]

    # Creating pseudo-labels for gyroscope data
    gyro_data['body_part'] = gyro_data['device'].map(device_to_body_part)
    gyro_data['pseudo_label'] = gyro_data['body_part'] + '_gyro'

    gyro_datasets = {}
    for user in har_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        labels = user_extract["pseudo_label"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    # Combining accelerometer and gyroscope data
    all_data = {}
    gyro_acc_data = pd.concat([acc_data, gyro_data])
    gyro_acc_data.dropna(how="any", inplace=True)

    for user in har_users:
        user_extract = gyro_acc_data[gyro_acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        labels = user_extract["pseudo_label"].values
        print(f"{user} {data.shape}")
        all_data[user] = [(data, labels)]

    user_datasets = {'acc': acc_datasets,
                     'gyro': gyro_datasets,
                     'all': all_data}
    print(f"hhar user_dataset: {user_datasets}")
    return user_datasets


# def process_hhar_all_files(data_folder_path):
#     """
#     Preprocess all files of from the HHAR dataset
#     Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
#     Args:
#         data_folder_path: folder path
#
#     Returns:
#         user_datasets (a list containing data collected from the acc + gyro from both smartphones and watches)
#     """
#     har_phone_acc = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
#     har_phone_gyro = pd.read_csv(os.path.join(data_folder_path, 'Phones_gyroscope.csv'))
#     har_watch_acc = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
#     har_watch_gyro = pd.read_csv(os.path.join(data_folder_path, 'Watch_gyroscope.csv'))
#     acc_data = pd.concat([har_phone_acc, har_watch_acc])
#     gyro_data = pd.concat([har_phone_gyro, har_watch_gyro])
#
#     acc_data.dropna(how="any", inplace=True)
#     acc_data = acc_data[['x', 'y', 'z', "gt", "User", "Device"]]
#     acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
#     har_users = acc_data["user-id"].unique()
#
#     acc_datasets = {}
#     for user in har_users:
#         user_extract = acc_data[acc_data["user-id"] == user]
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "HHAR"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         acc_datasets[user] = [(data, labels)]
#
#     gyro_data.dropna(how="any", inplace=True)
#     gyro_data = gyro_data[['x', 'y', 'z', "gt", "User", "Device"]]
#     gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
#     gyro_datasets = {}
#     for user in har_users:
#         user_extract = gyro_data[gyro_data["user-id"] == user]
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "HHAR"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         gyro_datasets[user] = [(data, labels)]
#
#     all_data = {}
#     gyro_acc_data = pd.concat([acc_data, gyro_data])
#     gyro_acc_data.dropna(how="any", inplace=True)
#
#     for user in har_users:
#         user_extract = gyro_acc_data[gyro_acc_data["user-id"] == user]
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "HHAR"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         all_data[user] = [(data, labels)]
#
#     user_datasets = {'acc': acc_datasets,
#                      'gyro': gyro_datasets,
#                      'all': all_data}
#     print(f"hhar user_dataset: {user_datasets}")
#     return user_datasets

def process_PAMAP2_all_data(data_folder_path):
    combined_data = []

    for filename in os.listdir(data_folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_folder_path, filename)
            df = pd.read_csv(file_path, sep=' ', header=None, na_filter="NaN")
            df['User-ID'] = filename[7:10]
            combined_data.append(df)

    user_df = pd.concat(combined_data, ignore_index=True)
    user_df.drop(user_df[user_df[1] == 0].index, inplace=True)

    # Mapping devices to body parts
    device_to_body_part = {
        'IMU hand acc': 'hand',
        'IMU hand gyro': 'hand',
        'IMU hand mag': 'hand',
        'IMU chest acc': 'chest',
        'IMU chest gyro': 'chest',
        'IMU chest mag': 'chest',
        'IMU ankle acc': 'ankle',
        'IMU ankle gyro': 'ankle',
        'IMU ankle mag': 'ankle'
    }

    def add_device_data(df, sensor_cols, device_name):
        data = df.loc[:, sensor_cols + [1, 'User-ID']].copy()
        data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
        data['device'] = device_name
        data['body_part'] = device_to_body_part[device_name]
        data['pseudo_label'] = data['body_part'] + '_' + device_name.split()[-1]
        return data

    hand_acc_data = add_device_data(user_df, [4, 5, 6], "IMU hand acc")
    hand_gyro_data = add_device_data(user_df, [10, 11, 12], "IMU hand gyro")
    hand_mag_data = add_device_data(user_df, [13, 14, 15], "IMU hand mag")
    chest_acc_data = add_device_data(user_df, [21, 22, 23], "IMU chest acc")
    chest_gyro_data = add_device_data(user_df, [27, 28, 29], "IMU chest gyro")
    chest_mag_data = add_device_data(user_df, [30, 31, 32], "IMU chest mag")
    ankle_acc_data = add_device_data(user_df, [38, 39, 40], "IMU ankle acc")
    ankle_gyro_data = add_device_data(user_df, [44, 45, 46], "IMU ankle gyro")
    ankle_mag_data = add_device_data(user_df, [47, 48, 49], "IMU ankle mag")

    PAMAP_users = user_df['User-ID'].unique()

    def create_dataset(data_parts, users):
        datasets = {}
        for user in users:
            user_extract = pd.concat(data_parts)
            user_extract = user_extract[user_extract["user-id"] == user].dropna()
            data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
            labels = user_extract["pseudo_label"].values
            datasets[user] = [(data, labels)]
        return datasets

    pamap_acc_datasets = create_dataset([hand_acc_data, chest_acc_data, ankle_acc_data], PAMAP_users)
    pamap_gyro_datasets = create_dataset([hand_gyro_data, chest_gyro_data, ankle_gyro_data], PAMAP_users)
    pamap_mag_datasets = create_dataset([hand_mag_data, chest_mag_data, ankle_mag_data], PAMAP_users)
    pamap_all_data = create_dataset([hand_acc_data, hand_gyro_data, hand_mag_data, chest_acc_data, chest_gyro_data, chest_mag_data, ankle_acc_data, ankle_gyro_data, ankle_mag_data], PAMAP_users)

    user_datasets = {
        'acc': pamap_acc_datasets,
        'gyro': pamap_gyro_datasets,
        'mag': pamap_mag_datasets,
        'all': pamap_all_data
    }
    print(f"PAMAP2 dataset :{user_datasets}")
    return user_datasets

# def process_PAMAP2_all_data(data_folder_path):
#     combined_data = []
#
#     for filename in os.listdir(data_folder_path):
#         if filename.endswith('.dat'):
#             file_path = os.path.join(data_folder_path, filename)
#             df = pd.read_csv(file_path, sep=' ', header=None, na_filter="NaN")
#             df['User-ID'] = filename[7:10]
#             combined_data.append(df)
#
#     user_df = pd.concat(combined_data, ignore_index=True)
#     user_df.drop(user_df[user_df[1] == 0].index, inplace=True)
#     hand_acc_data = user_df.loc[:, [4, 5, 6, 1, 'User-ID']]
#     hand_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     hand_acc_data['device'] = "IMU hand acc"
#
#     hand_gyro_data = user_df.loc[:, [10, 11, 12, 1, 'User-ID']]
#     hand_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     hand_gyro_data['device'] = 'IMU hand gyro'
#
#     hand_mag_data = user_df.loc[:, [13, 14, 15, 1, 'User-ID']]
#     hand_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     hand_mag_data['device'] = 'IMU hand mag'
#
#     chest_acc_data = user_df.loc[:, [21, 22, 23, 1, 'User-ID']]
#     chest_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     chest_acc_data['device'] = "IMU chest acc"
#
#     chest_gyro_data = user_df.loc[:, [27, 28, 29, 1, 'User-ID']]
#     chest_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     chest_gyro_data['device'] = "IMU chest gyro"
#
#     chest_mag_data = user_df.loc[:, [30, 31, 32, 1, 'User-ID']]
#     chest_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     chest_mag_data['device'] = "IMU chest mag"
#
#     ankle_acc_data = user_df.loc[:, [38, 39, 40, 1, 'User-ID']]
#     ankle_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     ankle_acc_data['device'] = 'IMU ankle acc'
#
#     ankle_gyro_data = user_df.loc[:, [44, 45, 46, 1, 'User-ID']]
#     ankle_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     ankle_gyro_data['device'] = 'IMU ankle gyro'
#
#     ankle_mag_data = user_df.loc[:, [47, 48, 49, 1, 'User-ID']]
#     ankle_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
#     ankle_mag_data['device'] = 'IMU ankle mag'
#
#     PAMAP_users = user_df['User-ID'].unique()
#
#     pamap_acc_data = pd.concat([hand_acc_data, chest_acc_data, ankle_acc_data])
#     pamap_acc_datasets = {}
#     for user in PAMAP_users:
#         user_extract = pamap_acc_data[pamap_acc_data["user-id"] == user]
#         user_extract = user_extract.dropna()
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "PAMAP"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         pamap_acc_datasets[user] = [(data, labels)]
#     pamap_gyro_data = pd.concat([hand_gyro_data, chest_gyro_data, ankle_gyro_data])
#     pamap_gyro_datasets = {}
#     for user in PAMAP_users:
#         user_extract = pamap_gyro_data[pamap_gyro_data["user-id"] == user]
#         user_extract = user_extract.dropna()
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "PAMAP"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         pamap_gyro_datasets[user] = [(data, labels)]
#
#     pamap_mag_data = pd.concat([hand_mag_data, chest_mag_data, ankle_mag_data])
#     pamap_mag_datasets = {}
#     for user in PAMAP_users:
#         user_extract = pamap_mag_data[pamap_mag_data["user-id"] == user]
#         user_extract = user_extract.dropna()
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "PAMAP"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         pamap_mag_datasets[user] = [(data, labels)]
#
#     pamap_all_data = {}
#     gyro_acc_mag_data = pd.concat([pamap_acc_data, pamap_gyro_data, pamap_mag_data])
#     for user in PAMAP_users:
#         user_extract = gyro_acc_mag_data[gyro_acc_mag_data["user-id"] == user]
#         user_extract = user_extract.dropna()
#         data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
#         # data["data-source"] = "PAMAP"
#         labels = user_extract["device"].values
#         print(f"{user} {data.shape}")
#         pamap_all_data[user] = [(data, labels)]
#
#     user_datasets = {'acc': pamap_acc_datasets,
#                      'gyro': pamap_gyro_datasets,
#                      'mag': pamap_mag_datasets,
#                      'all': pamap_all_data}
#     print(f"PAMAP2 dataset: {user_datasets}")
#     return user_datasets


def concat_datasets(datasets, sensor_type):
    concated_datasets = {}
    ##TODO need to check key to see whether mag exists
    for df in datasets:
        concated_datasets.update(df[sensor_type])
    return concated_datasets


def process_motion_sense_all_files(data_folder_path):
    user_datasets = {}
    all_sensor_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = pd.DataFrame()
    for folder in all_sensor_folders:
        # print(folder)
        print(folder[46:49])
        all_trials_folders = sorted(glob.glob(folder + "/*"))
        df = pd.DataFrame()

        for trial_folder in all_trials_folders:
            sensor_type = folder[46:49]
            # label of the trial is given in the folder name, separated by underscore
            label = "iphone_" + sensor_type.lower()
            # label_set[label] = True
            for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):
                user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
                if user_id_match is not None:
                    user_id = int(user_id_match.group('user_id'))
                    user_trial_dataset = pd.read_csv(trial_user_file)
                    table = user_trial_dataset[['x', 'y', 'z']]
                    table['device'] = label
                    table['User-ID'] = str(user_id)
                    df = pd.concat([df, table])
                    all_data = pd.concat([all_data, table])
                else:
                    print("[ERR] User id not found", trial_user_file)
        sensor_datasets = {}
        ms_users = df['User-ID'].unique()
        for user in ms_users:
            user_extract = df[df['User-ID'] == user]
            data = user_extract[['x', 'y', 'z']].copy()
            labels = user_extract['device']
            sensor_datasets[user] = [(data, labels)]

        if (folder[46:49] == "Acc"):
            user_datasets.update({'acc': sensor_datasets})
        else:
            user_datasets.update({'gyro': sensor_datasets})
    sensor_datasets = {}
    ms_users = all_data['User-ID'].unique()
    for user in ms_users:
        user_extract = all_data[all_data['User-ID'] == user]
        data = user_extract[['x', 'y', 'z']].copy()
        labels = user_extract['device']
        sensor_datasets[user] = [(data, labels)]
    user_datasets.update({'all': sensor_datasets})
    print(f"motionsense dataset: {user_datasets}")
    return user_datasets


def process_pamap2_har_files(data_folder_path):
    combined_data = []
    map = {0: 'transient', 1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling',
           7: 'Nordic_walking', 9: 'watching_TV', 10: 'computer_work', 11: 'car driving', 12: 'ascending_stairs',
           13: 'descending_stairs', 16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry', 19: 'house_cleaning',
           20: 'playing_soccer', 24: 'rope_jumping'}

    for filename in os.listdir(data_folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_folder_path, filename)
            df = pd.read_csv(file_path, sep=' ', header=None, na_filter="NaN")
            df['User-ID'] = filename[7:10]
            combined_data.append(df)
    print(f"combined data: {combined_data}")
    user_df = pd.concat(combined_data, ignore_index=True)
    user_df.drop(user_df[user_df[1] == 0].index, inplace=True)
    print(user_df.head())
    user_df[1].replace(map, inplace=True)
    hand_acc_data = user_df.loc[:, [4, 5, 6, 1, 'User-ID']]
    hand_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_acc_data['device'] = "IMU hand acc"

    hand_gyro_data = user_df.loc[:, [10, 11, 12, 1, 'User-ID']]
    hand_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_gyro_data['device'] = 'IMU hand gyro'

    hand_mag_data = user_df.loc[:, [13, 14, 15, 1, 'User-ID']]
    hand_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_mag_data['device'] = 'IMU hand mag'

    chest_acc_data = user_df.loc[:, [21, 22, 23, 1, 'User-ID']]
    chest_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_acc_data['device'] = "IMU chest acc"

    chest_gyro_data = user_df.loc[:, [27, 28, 29, 1, 'User-ID']]
    chest_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_gyro_data['device'] = "IMU chest gyro"

    chest_mag_data = user_df.loc[:, [30, 31, 32, 1, 'User-ID']]
    chest_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_mag_data['device'] = "IMU chest mag"

    ankle_acc_data = user_df.loc[:, [38, 39, 40, 1, 'User-ID']]
    ankle_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_acc_data['device'] = 'IMU ankle acc'

    ankle_gyro_data = user_df.loc[:, [44, 45, 46, 1, 'User-ID']]
    ankle_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_gyro_data['device'] = 'IMU ankle gyro'

    ankle_mag_data = user_df.loc[:, [47, 48, 49, 1, 'User-ID']]
    ankle_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_mag_data['device'] = 'IMU ankle mag'

    PAMAP_users = user_df['User-ID'].unique()

    pamap_acc_data = pd.concat([hand_acc_data, chest_acc_data, ankle_acc_data])
    pamap_acc_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_acc_data[pamap_acc_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_acc_datasets[user] = [(data, labels)]
    pamap_gyro_data = pd.concat([hand_gyro_data, chest_gyro_data, ankle_gyro_data])
    pamap_gyro_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_gyro_data[pamap_gyro_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_gyro_datasets[user] = [(data, labels)]

    pamap_mag_data = pd.concat([hand_mag_data, chest_mag_data, ankle_mag_data])
    pamap_mag_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_mag_data[pamap_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_mag_datasets[user] = [(data, labels)]

    pamap_all_data = {}
    gyro_acc_mag_data = pd.concat([pamap_acc_data, pamap_gyro_data, pamap_mag_data])
    for user in PAMAP_users:
        user_extract = gyro_acc_mag_data[gyro_acc_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_all_data[user] = [(data, labels)]

    user_datasets = {'acc': pamap_acc_datasets,
                     'gyro': pamap_gyro_datasets,
                     'mag': pamap_mag_datasets,
                     'all': pamap_all_data}
    return user_datasets


def process_hhar_all_har_files(data_folder_path):
    """
    Preprocess all files of from the HHAR dataset
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    Args:
        data_folder_path: folder path

    Returns:
        user_datasets (a list containing data collected from the acc + gyro from both smartphones and watches)
    """
    har_phone_acc = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    har_phone_acc['Device'] = "Phone acc"
    har_phone_gyro = pd.read_csv(os.path.join(data_folder_path, 'Phones_gyroscope.csv'))
    har_phone_gyro['Device'] = "Phone gyro"
    har_watch_acc = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
    har_watch_acc['Device'] = "Watch acc"
    har_watch_gyro = pd.read_csv(os.path.join(data_folder_path, 'Watch_gyroscope.csv'))
    har_watch_gyro['Device'] = "Watch gyro"
    acc_data = pd.concat([har_phone_acc, har_watch_acc])
    gyro_data = pd.concat([har_phone_gyro, har_watch_gyro])

    acc_data.dropna(how="any", inplace=True)
    acc_data = acc_data[['x', 'y', 'z', "gt", "User", "Device"]]
    acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    har_users = acc_data["user-id"].unique()

    acc_datasets = {}
    for user in har_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    gyro_data.dropna(how="any", inplace=True)
    gyro_data = gyro_data[['x', 'y', 'z', "gt", "User", "Device"]]
    gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    gyro_datasets = {}
    for user in har_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    all_data = {}
    gyro_acc_data = pd.concat([acc_data, gyro_data])
    gyro_acc_data.dropna(how="any", inplace=True)

    for user in har_users:
        user_extract = gyro_acc_data[gyro_acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        all_data[user] = [(data, labels)]

    user_datasets = {'acc': acc_datasets,
                     'gyro': gyro_datasets,
                     'all': all_data}
    # print(f"hhar har dataset: {user_datasets}")
    return user_datasets


def process_motion_sense_all_har_files(data_folder_path):
    user_datasets = {}
    all_sensor_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = pd.DataFrame()
    for folder in all_sensor_folders:
        # print(folder)
        print(folder[46:49])
        all_trials_folders = sorted(glob.glob(folder + "/*"))
        df = pd.DataFrame()

        for trial_folder in all_trials_folders:
            trial_name = os.path.split(trial_folder)[-1]

            # label of the trial is given in the folder name, separated by underscore
            label = trial_name.split("_")[0]
            # label_set[label] = True
            for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):
                user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
                if user_id_match is not None:
                    user_id = int(user_id_match.group('user_id'))
                    user_trial_dataset = pd.read_csv(trial_user_file)
                    table = user_trial_dataset[['x', 'y', 'z']]
                    table['activity'] = label
                    table['User-ID'] = str(user_id)
                    df = pd.concat([df, table])
                    all_data = pd.concat([all_data, table])
                else:
                    print("[ERR] User id not found", trial_user_file)
        sensor_datasets = {}
        ms_users = df['User-ID'].unique()
        for user in ms_users:
            user_extract = df[df['User-ID'] == user]
            data = user_extract[['x', 'y', 'z']].copy()
            labels = user_extract['activity']
            sensor_datasets[user] = [(data, labels)]
        if (folder[46:49] == "Acc"):
            user_datasets.update({'acc': sensor_datasets})
        else:
            user_datasets.update({'gyro': sensor_datasets})

    sensor_datasets = {}
    ms_users = all_data['User-ID'].unique()
    for user in ms_users:
        user_extract = all_data[all_data['User-ID'] == user]
        data = user_extract[['x', 'y', 'z']].copy()
        labels = user_extract['activity']
        sensor_datasets[user] = [(data, labels)]
    user_datasets.update({'all': sensor_datasets})
    # print(f"motionsense: {user_datasets}")
    return user_datasets


def process_HARTH_thigh_acc_har_data(data_folder_path):
    combined_data = []
    for filename in os.listdir(data_folder_path):
        file_path = os.path.join(data_folder_path, filename)
        df = pd.read_csv(file_path, sep=',', header=0, na_filter="NaN")
        df['User-ID'] = filename[1:4]
        combined_data.append(df)
    df = pd.concat(combined_data)
    activity_map = {
        1: "walking",
        2: "running",
        3: "shuffling",
        4: "stairs (ascending)",
        5: "stairs (descending)",
        6: "standing",
        7: "sitting",
        8: "lying",
        13: "cycling (sit)",
        14: "cycling (stand)",
        130: "cycling (sit, inactive)",
        140: "cycling (stand, inactive)"
    }
    df['label'].replace(activity_map, inplace=True)
    back_data = df.loc[:, ['User-ID', 'x-axis', 'y-axis', 'z-axis', 'label']]
    harth_user = back_data["User-ID"].unique()

    user_datasets = {}
    for user in harth_user:
        user_extract = back_data[back_data['User-ID'] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']].values
        labels = user_extract["label"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data, labels)]
    userdataset = {"acc": user_datasets,
                   "all": user_datasets}
    return userdataset


def process_HARTH_back_acc_har_data(data_folder_path):
    combined_data = []
    for filename in os.listdir(data_folder_path):
        file_path = os.path.join(data_folder_path, filename)
        df = pd.read_csv(file_path, sep=',', header=0, na_filter="NaN")
        df['User-ID'] = filename[1:4]
        combined_data.append(df)
    df = pd.concat(combined_data)
    activity_map = {
        1: "walking",
        2: "running",
        3: "shuffling",
        4: "stairs (ascending)",
        5: "stairs (descending)",
        6: "standing",
        7: "sitting",
        8: "lying",
        13: "cycling (sit)",
        14: "cycling (stand)",
        130: "cycling (sit, inactive)",
        140: "cycling (stand, inactive)"
    }
    df['label'].replace(activity_map, inplace=True)
    back_data = df.loc[:, ['User-ID', 'x-axis', 'y-axis', 'z-axis' 'label']]
    harth_user = back_data["User-ID"].unique()

    user_datasets = {}
    for user in harth_user:
        user_extract = back_data[back_data['User-ID'] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']].values
        labels = user_extract["label"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data, labels)]
    userdataset = {"acc": user_datasets,
                   "all": user_datasets}
    return userdataset


def process_HARTH_all_har_data(data_folder_path):
    combined_data = []
    for filename in os.listdir(data_folder_path):
        file_path = os.path.join(data_folder_path, filename)
        df = pd.read_csv(file_path, sep=',', header=0, na_filter="NaN")
        df["User-ID"] = filename[1:4]
        combined_data.append(df)
    df = pd.concat(combined_data)
    back_df = df.loc[:, ['User-ID', 'back_x', 'back_y', 'back_z', 'label']]
    back_df.columns = ['User-ID', 'x-axis', 'y-axis', 'z-axis', 'label']
    thigh_df = df.loc[:, ['User-ID', 'thigh_x', 'thigh_y', 'thigh_z', 'label']]
    thigh_df.columns = ['User-ID', 'x-axis', 'y-axis', 'z-axis', 'label']
    df = pd.concat([back_df, thigh_df])
    activity_map = {
        1: "walking",
        2: "running",
        3: "shuffling",
        4: "stairs (ascending)",
        5: "stairs (descending)",
        6: "standing",
        7: "sitting",
        8: "lying",
        13: "cycling (sit)",
        14: "cycling (stand)",
        130: "cycling (sit, inactive)",
        140: "cycling (stand, inactive)"
    }
    df['label'].replace(activity_map, inplace=True)
    harth_user = df['User-ID'].unique()
    datasets = {}
    for user in harth_user:
        user_extract = df[df['User-ID'] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["label"]
        print(f"{user} {data.shape}")
        datasets[user] = [(data, labels)]
    userdataset = {"acc": datasets,
                   "all": datasets}
    return userdataset


def process_HARTH_all_acc_data(data_folder_path):
    combined_data = []
    for filename in os.listdir(data_folder_path):
        file_path = os.path.join(data_folder_path, filename)
        df = pd.read_csv(file_path, sep=',', header=0, na_filter="NaN")
        df["User-ID"] = filename[1:4]
        combined_data.append(df)
    df = pd.concat(combined_data)
    back_df = df.loc[:, ['User-ID', 'back_x', 'back_y', 'back_z']]
    back_df.columns = ['User-ID', 'x-axis', 'y-axis', 'z-axis']
    back_df['device'] = "back_acc"
    thigh_df = df.loc[:, ['User-ID', 'thigh_x', 'thigh_y', 'thigh_z']]
    thigh_df.columns = ['User-ID', 'x-axis', 'y-axis', 'z-axis']
    thigh_df['device'] = "thigh_acc"
    df = pd.concat([back_df, thigh_df])
    harth_user = df['User-ID'].unique()
    datasets = {}
    for user in harth_user:
        user_extract = df[df['User-ID'] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        print(f"{user} {data.shape}")
        datasets[user] = [(data, labels)]
    user_datasets = {"acc": datasets,
                     "all": datasets}
    print(f"HARTH dataset: {user_datasets}")
    return user_datasets


def process_dasa_all_data_standardised(data_folder_path):
    user_datasets = {}
    all_activity_folders = sorted(glob.glob(data_folder_path + "/*"))
    df = pd.DataFrame()
    for folder in all_activity_folders:
        # print(folder[32:])
        all_user_folders = sorted(glob.glob(folder + "/*"))

        for user_folder in all_user_folders:
            # print(user_folder[37:])
            user_id = user_folder[37:]
            user_segments = sorted(glob.glob(user_folder + "/*"))
            segment_data = []
            for segments in user_segments:
                temp = pd.read_csv(segments, header=None)
                temp['user-id'] = user_id
                temp['activity'] = folder[32:]
                segment_data.append(temp)
            df = pd.concat([df, temp], axis=0)
    torso_acc_data = df.loc[:, [0, 1, 2, 'activity', 'user-id']]
    torso_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_acc_data['device'] = 'torso acc'

    torso_gyro_data = df.loc[:, [3, 4, 5, 'activity', 'user-id']]
    torso_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_gyro_data['device'] = 'torso gyro'

    torso_mag_data = df.loc[:, [6, 7, 8, 'activity', 'user-id']]
    torso_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_mag_data['device'] = 'torso mag'

    ra_acc_data = df.loc[:, [9, 10, 11, 'activity', 'user-id']]
    ra_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_acc_data['device'] = 'right arm acc'

    ra_gyro_data = df.loc[:, [12, 13, 14, 'activity', 'user-id']]
    ra_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_gyro_data['device'] = 'right arm gyro'

    ra_mag_data = df.loc[:, [15, 16, 17, 'activity', 'user-id']]
    ra_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_mag_data['device'] = 'right arm mag'

    la_acc_data = df.loc[:, [18, 19, 20, 'activity', 'user-id']]
    la_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_acc_data['device'] = 'left arm acc'

    la_gyro_data = df.loc[:, [21, 22, 23, 'activity', 'user-id']]
    la_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_gyro_data['device'] = 'left arm gyro'

    la_mag_data = df.loc[:, [24, 25, 26, 'activity', 'user-id']]
    la_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_mag_data['device'] = 'left arm mag'

    rl_acc_data = df.loc[:, [27, 28, 29, 'activity', 'user-id']]
    rl_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_acc_data['device'] = 'right leg acc'

    rl_gyro_data = df.loc[:, [30, 31, 32, 'activity', 'user-id']]
    rl_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_gyro_data['device'] = 'right leg gyro'

    rl_mag_data = df.loc[:, [33, 34, 35, 'activity', 'user-id']]
    rl_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_mag_data['device'] = 'right leg mag'

    ll_acc_data = df.loc[:, [36, 37, 38, 'activity', 'user-id']]
    ll_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_acc_data['device'] = 'left leg acc'

    ll_gyro_data = df.loc[:, [39, 40, 41, 'activity', 'user-id']]
    ll_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_gyro_data['device'] = 'left leg gyro'

    ll_mag_data = df.loc[:, [42, 43, 44, 'activity', 'user-id']]
    ll_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_mag_data['device'] = 'left leg mag'

    all_users = df['user-id'].unique()
    acc_data = pd.concat([torso_acc_data, rl_acc_data, ll_acc_data, ra_acc_data, rl_acc_data], ignore_index=True)
    acc_data = normalize_sampling_rate(acc_data, '40L')
    acc_datasets = {}
    for user in all_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]
    print('done acc')
    gyro_data = pd.concat([torso_gyro_data, rl_gyro_data, ll_gyro_data, ra_gyro_data, rl_gyro_data], ignore_index=True)
    gyro_data = normalize_sampling_rate(gyro_data, '40L')
    gyro_datasets = {}
    for user in all_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    mag_data = pd.concat([torso_mag_data, rl_mag_data, ll_mag_data, ra_mag_data, rl_mag_data], ignore_index=True)
    mag_data = normalize_sampling_rate(mag_data, '40L')
    mag_datasets = {}
    for user in all_users:
        user_extract = mag_data[mag_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        mag_datasets[user] = [(data, labels)]

    all_data = pd.concat([acc_data, gyro_data, mag_data])
    all_datasets = {}
    for user in all_users:
        user_extract = all_data[all_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        all_datasets[user] = [(data, labels)]

    user_datasets = {
        'acc': acc_datasets,
        'gyro': gyro_datasets,
        'mag': mag_datasets,
        'all': all_datasets
    }
    return user_datasets


def process_dasa_all_har_data_standardised(data_folder_path):
    user_datasets = {}
    all_activity_folders = sorted(glob.glob(data_folder_path + "/*"))
    df = pd.DataFrame()
    for folder in all_activity_folders:
        # print(folder[32:])
        all_user_folders = sorted(glob.glob(folder + "/*"))

        for user_folder in all_user_folders:
            # print(user_folder[37:])
            user_id = user_folder[37:]
            user_segments = sorted(glob.glob(user_folder + "/*"))
            segment_data = []
            for segments in user_segments:
                temp = pd.read_csv(segments, header=None)
                temp['user-id'] = user_id
                temp['activity'] = folder[32:]
                segment_data.append(temp)
            df = pd.concat([df, temp], axis=0)
    activity_map = {
        "a01": "sitting",
        "a02": "standing",
        "a03": "lying on back",
        "a04": "lying on right side",
        "a05": "ascending stairs",
        "a06": "descending stairs",
        "a07": "standing in still elevator",
        "a08": "moving around in elevator",
        "a09": "walking in parking lot",
        "a10": "walking on treadmill (flat)",
        "a11": "walking on treadmill (15 deg inclined)",
        "a12": "running on treadmill (8 km/h)",
        "a13": "exercising on stepper",
        "a14": "exercising on cross trainer",
        "a15": "cycling on exercise bike (horizontal)",
        "a16": "cycling on exercise bike (vertical)",
        "a17": "rowing",
        "a18": "jumping",
        "a19": "playing basketball"
    }
    df['activity'].replace(activity_map, inplace=True)
    torso_acc_data = df.loc[:, [0, 1, 2, 'activity', 'user-id']]
    torso_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_acc_data['device'] = 'torso acc'

    torso_gyro_data = df.loc[:, [3, 4, 5, 'activity', 'user-id']]
    torso_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_gyro_data['device'] = 'torso gyro'

    torso_mag_data = df.loc[:, [6, 7, 8, 'activity', 'user-id']]
    torso_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_mag_data['device'] = 'torso mag'

    ra_acc_data = df.loc[:, [9, 10, 11, 'activity', 'user-id']]
    ra_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_acc_data['device'] = 'right arm acc'

    ra_gyro_data = df.loc[:, [12, 13, 14, 'activity', 'user-id']]
    ra_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_gyro_data['device'] = 'right arm gyro'

    ra_mag_data = df.loc[:, [15, 16, 17, 'activity', 'user-id']]
    ra_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_mag_data['device'] = 'right arm mag'

    la_acc_data = df.loc[:, [18, 19, 20, 'activity', 'user-id']]
    la_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_acc_data['device'] = 'left arm acc'

    la_gyro_data = df.loc[:, [21, 22, 23, 'activity', 'user-id']]
    la_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_gyro_data['device'] = 'left arm gyro'

    la_mag_data = df.loc[:, [24, 25, 26, 'activity', 'user-id']]
    la_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_mag_data['device'] = 'left arm mag'

    rl_acc_data = df.loc[:, [27, 28, 29, 'activity', 'user-id']]
    rl_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_acc_data['device'] = 'right leg acc'

    rl_gyro_data = df.loc[:, [30, 31, 32, 'activity', 'user-id']]
    rl_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_gyro_data['device'] = 'right leg gyro'

    rl_mag_data = df.loc[:, [33, 34, 35, 'activity', 'user-id']]
    rl_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_mag_data['device'] = 'right leg mag'

    ll_acc_data = df.loc[:, [36, 37, 38, 'activity', 'user-id']]
    ll_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_acc_data['device'] = 'left leg acc'

    ll_gyro_data = df.loc[:, [39, 40, 41, 'activity', 'user-id']]
    ll_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_gyro_data['device'] = 'left leg gyro'

    ll_mag_data = df.loc[:, [42, 43, 44, 'activity', 'user-id']]
    ll_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_mag_data['device'] = 'left leg mag'

    all_users = df['user-id'].unique()
    acc_data = pd.concat([torso_acc_data, rl_acc_data, ll_acc_data, ra_acc_data, rl_acc_data], ignore_index=True)
    acc_data = normalize_sampling_rate(acc_data, '40L')
    acc_datasets = {}
    for user in all_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    gyro_data = pd.concat([torso_gyro_data, rl_gyro_data, ll_gyro_data, ra_gyro_data, rl_gyro_data], ignore_index=True)
    gyro_data = normalize_sampling_rate(gyro_data, '40L')

    gyro_datasets = {}
    for user in all_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    mag_data = pd.concat([torso_mag_data, rl_mag_data, ll_mag_data, ra_mag_data, rl_mag_data], ignore_index=True)
    mag_data = normalize_sampling_rate(mag_data, '40L')
    mag_datasets = {}
    for user in all_users:
        user_extract = mag_data[mag_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        mag_datasets[user] = [(data, labels)]

    all_data = pd.concat([acc_data, gyro_data, mag_data])
    all_datasets = {}
    for user in all_users:
        user_extract = all_data[all_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        all_datasets[user] = [(data, labels)]

    user_datasets = {
        'acc': acc_datasets,
        'gyro': gyro_datasets,
        'mag': mag_datasets,
        'all': all_datasets
    }

    return user_datasets


def process_dasa_all_data(data_folder_path):
    user_datasets = {}
    all_activity_folders = sorted(glob.glob(data_folder_path + "/*"))
    df = pd.DataFrame()

    for folder in all_activity_folders:
        all_user_folders = sorted(glob.glob(folder + "/*"))
        for user_folder in all_user_folders:
            user_id = user_folder[37:]
            user_segments = sorted(glob.glob(user_folder + "/*"))
            for segments in user_segments:
                temp = pd.read_csv(segments, header=None)
                temp['user-id'] = user_id
                temp['activity'] = folder[32:]
                df = pd.concat([df, temp], axis=0)

    # Mapping device to pseudo labels
    device_to_pseudo_label = {
        'torso acc': 'torso-acc',
        'torso gyro': 'torso-gyro',
        'torso mag': 'torso-mag',
        'right arm acc': 'right_arm-acc',
        'right arm gyro': 'right_arm-gyro',
        'right arm mag': 'right_arm-mag',
        'left arm acc': 'left_arm-acc',
        'left arm gyro': 'left_arm-gyro',
        'left arm mag': 'left_arm-mag',
        'right leg acc': 'right_leg-acc',
        'right leg gyro': 'right_leg-gyro',
        'right leg mag': 'right_leg-mag',
        'left leg acc': 'left_leg-acc',
        'left leg gyro': 'left_leg-gyro',
        'left leg mag': 'left_leg-mag'
    }

    def add_device_data(df, cols, device_name):
        data = df.loc[:, ['user-id'] + cols + ['activity']].copy()
        data.columns = ['user-id', 'x-axis', 'y-axis', 'z-axis', 'activity']
        data['device'] = device_name
        data['pseudo_label'] = device_to_pseudo_label[device_name]
        return data

    torso_acc_data = add_device_data(df, [0, 1, 2], 'torso acc')
    torso_gyro_data = add_device_data(df, [3, 4, 5], 'torso gyro')
    torso_mag_data = add_device_data(df, [6, 7, 8], 'torso mag')
    ra_acc_data = add_device_data(df, [9, 10, 11], 'right arm acc')
    ra_gyro_data = add_device_data(df, [12, 13, 14], 'right arm gyro')
    ra_mag_data = add_device_data(df, [15, 16, 17], 'right arm mag')
    la_acc_data = add_device_data(df, [18, 19, 20], 'left arm acc')
    la_gyro_data = add_device_data(df, [21, 22, 23], 'left arm gyro')
    la_mag_data = add_device_data(df, [24, 25, 26], 'left arm mag')
    rl_acc_data = add_device_data(df, [27, 28, 29], 'right leg acc')
    rl_gyro_data = add_device_data(df, [30, 31, 32], 'right leg gyro')
    rl_mag_data = add_device_data(df, [33, 34, 35], 'right leg mag')
    ll_acc_data = add_device_data(df, [36, 37, 38], 'left leg acc')
    ll_gyro_data = add_device_data(df, [39, 40, 41], 'left leg gyro')
    ll_mag_data = add_device_data(df, [42, 43, 44], 'left leg mag')

    acc_data = pd.concat([torso_acc_data, rl_acc_data, ll_acc_data, ra_acc_data, la_acc_data], ignore_index=True)
    gyro_data = pd.concat([torso_gyro_data, rl_gyro_data, ll_gyro_data, ra_gyro_data, la_gyro_data], ignore_index=True)
    mag_data = pd.concat([torso_mag_data, rl_mag_data, ll_mag_data, ra_mag_data, la_mag_data], ignore_index=True)

    all_users = df['user-id'].unique()

    acc_datasets = {}
    for user in all_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["pseudo_label"].values
        print(f"User {user} ACC data shape: {data.shape}")
        acc_datasets[user] = [(data, labels)]

    gyro_datasets = {}
    for user in all_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["pseudo_label"].values
        print(f"User {user} GYRO data shape: {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    mag_datasets = {}
    for user in all_users:
        user_extract = mag_data[mag_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["pseudo_label"].values
        print(f"User {user} MAG data shape: {data.shape}")
        mag_datasets[user] = [(data, labels)]

    all_data = pd.concat([acc_data, gyro_data, mag_data])
    all_datasets = {}
    for user in all_users:
        user_extract = all_data[all_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["pseudo_label"].values
        print(f"User {user} ALL data shape: {data.shape}")
        all_datasets[user] = [(data, labels)]

    user_datasets = {
        'acc': acc_datasets,
        'gyro': gyro_datasets,
        'mag': mag_datasets,
        'all': all_datasets
    }

    # Print unique pseudo labels
    unique_pseudo_labels = all_data['pseudo_label'].unique()
    print(f"Unique pseudo labels: {unique_pseudo_labels}")

    return user_datasets


def process_dasa_all_har_data(data_folder_path):
    user_datasets = {}
    all_activity_folders = sorted(glob.glob(data_folder_path + "/*"))
    df = pd.DataFrame()
    for folder in all_activity_folders:
        # print(folder[32:])
        all_user_folders = sorted(glob.glob(folder + "/*"))

        for user_folder in all_user_folders:
            # print(user_folder[37:])
            user_id = user_folder[37:]
            user_segments = sorted(glob.glob(user_folder + "/*"))
            segment_data = []
            for segments in user_segments:
                temp = pd.read_csv(segments, header=None)
                temp['user-id'] = user_id
                temp['activity'] = folder[32:]
                segment_data.append(temp)
            df = pd.concat([df, temp], axis=0)
    # activity_map = {
    #     "a01": "sitting",
    #     "a02": "standing",
    #     "a03": "lying on back",
    #     "a04": "lying on right side",
    #     "a05": "ascending stairs",
    #     "a06": "descending stairs",
    #     "a07": "standing in still elevator",
    #     "a08": "moving around in elevator",
    #     "a09": "walking in parking lot",
    #     "a10": "walking on treadmill (flat)",
    #     "a11": "walking on treadmill (15 deg inclined)",
    #     "a12": "running on treadmill (8 km/h)",
    #     "a13": "exercising on stepper",
    #     "a14": "exercising on cross trainer",
    #     "a15": "cycling on exercise bike (horizontal)",
    #     "a16": "cycling on exercise bike (vertical)",
    #     "a17": "rowing",
    #     "a18": "jumping",
    #     "a19": "playing basketball"
    # }
    # df['activity'].replace(activity_map, inplace=True)
    torso_acc_data = df.loc[:, [0, 1, 2, 'activity', 'user-id']]
    torso_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_acc_data['device'] = 'torso acc'

    torso_gyro_data = df.loc[:, [3, 4, 5, 'activity', 'user-id']]
    torso_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_gyro_data['device'] = 'torso gyro'

    torso_mag_data = df.loc[:, [6, 7, 8, 'activity', 'user-id']]
    torso_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    torso_mag_data['device'] = 'torso mag'

    ra_acc_data = df.loc[:, [9, 10, 11, 'activity', 'user-id']]
    ra_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_acc_data['device'] = 'right arm acc'

    ra_gyro_data = df.loc[:, [12, 13, 14, 'activity', 'user-id']]
    ra_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_gyro_data['device'] = 'right arm gyro'

    ra_mag_data = df.loc[:, [15, 16, 17, 'activity', 'user-id']]
    ra_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ra_mag_data['device'] = 'right arm mag'

    la_acc_data = df.loc[:, [18, 19, 20, 'activity', 'user-id']]
    la_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_acc_data['device'] = 'left arm acc'

    la_gyro_data = df.loc[:, [21, 22, 23, 'activity', 'user-id']]
    la_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_gyro_data['device'] = 'left arm gyro'

    la_mag_data = df.loc[:, [24, 25, 26, 'activity', 'user-id']]
    la_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    la_mag_data['device'] = 'left arm mag'

    rl_acc_data = df.loc[:, [27, 28, 29, 'activity', 'user-id']]
    rl_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_acc_data['device'] = 'right leg acc'

    rl_gyro_data = df.loc[:, [30, 31, 32, 'activity', 'user-id']]
    rl_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_gyro_data['device'] = 'right leg gyro'

    rl_mag_data = df.loc[:, [33, 34, 35, 'activity', 'user-id']]
    rl_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    rl_mag_data['device'] = 'right leg mag'

    ll_acc_data = df.loc[:, [36, 37, 38, 'activity', 'user-id']]
    ll_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_acc_data['device'] = 'left leg acc'

    ll_gyro_data = df.loc[:, [39, 40, 41, 'activity', 'user-id']]
    ll_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_gyro_data['device'] = 'left leg gyro'

    ll_mag_data = df.loc[:, [42, 43, 44, 'activity', 'user-id']]
    ll_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ll_mag_data['device'] = 'left leg mag'

    all_users = df['user-id'].unique()
    acc_data = pd.concat([torso_acc_data, rl_acc_data, ll_acc_data, ra_acc_data, rl_acc_data], ignore_index=True)
    acc_datasets = {}
    for user in all_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    gyro_data = pd.concat([torso_gyro_data, rl_gyro_data, ll_gyro_data, ra_gyro_data, rl_gyro_data], ignore_index=True)
    gyro_datasets = {}
    for user in all_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    mag_data = pd.concat([torso_mag_data, rl_mag_data, ll_mag_data, ra_mag_data, rl_mag_data], ignore_index=True)
    mag_datasets = {}
    for user in all_users:
        user_extract = mag_data[mag_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        mag_datasets[user] = [(data, labels)]

    all_data = pd.concat([acc_data, gyro_data, mag_data])
    all_datasets = {}
    for user in all_users:
        user_extract = all_data[all_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]]
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        all_datasets[user] = [(data, labels)]

    user_datasets = {
        'acc': acc_datasets,
        'gyro': gyro_datasets,
        'mag': mag_datasets,
        'all': all_datasets
    }
    # print(f"DSA dataset har: {user_datasets}")
    return user_datasets


def process_WISDM_all_har_data(data_folder_path):
    map = {
        'A': 'walking',
        'B': 'jogging',
        'C': 'stairs',
        'D': 'sitting',
        'E': 'standing',
        'F': 'typing',
        'G': 'teeth',
        'H': 'soup',
        'I': 'chips',
        'J': 'pasta',
        'K': 'drinking',
        'L': 'sandwich',
        'M': 'kicking',
        'O': 'catch',
        'P': 'dribbling',
        'Q': 'writing',
        'R': 'clapping',
        'S': 'folding'
    }

    all_device_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = []
    acc_data = []
    gyro_data = []
    for folder in all_device_folders:
        device = os.path.split(folder)[-1]
        all_sensor_folders = sorted(glob.glob(folder + "/*"))
        for sensor_folder in all_sensor_folders:
            sensor = os.path.split(sensor_folder)[-1]
            label = device + " " + sensor
            print(label)
            for trial_user_file in sorted(glob.glob(sensor_folder + "/*.txt")):
                df = pd.read_csv(trial_user_file, header=None)
                df[1].replace(map, inplace=True)
                df = df[[0, 1, 3, 4, 5]]
                df.columns = ['user-id', 'activity', 'x-axis', 'y-axis', 'z-axis']
                df['z-axis'].replace({';': ''}, regex=True, inplace=True)
                df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')
                df = df.astype({'user-id': 'string'})
                df['device'] = label
                all_data.append(df)
                if sensor == 'accel':
                    acc_data.append(df)
                elif sensor == 'gyro':
                    gyro_data.append(df)
    all_df = pd.concat(all_data)
    acc_df = pd.concat(acc_data)
    gyro_df = pd.concat(gyro_data)

    all_users = all_df['user-id'].unique()
    acc_dataset = {}
    for user in all_users:
        user_extract = acc_df[acc_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        acc_dataset[user] = [(data, labels)]

    gyro_dataset = {}
    for user in all_users:
        user_extract = gyro_df[gyro_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        gyro_dataset[user] = [(data, labels)]

    all_dataset = {}
    for user in all_users:
        user_extract = all_df[all_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        all_dataset[user] = [(data, labels)]

    user_dataset = {
        'acc': acc_dataset,
        'gyro': gyro_dataset,
        'all': all_dataset
    }
    return user_dataset


def process_WISDM_all_data(data_folder_path):
    all_device_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = []
    acc_data = []
    gyro_data = []
    for folder in all_device_folders:
        device = os.path.split(folder)[-1]
        all_sensor_folders = sorted(glob.glob(folder + "/*"))
        for sensor_folder in all_sensor_folders:
            sensor = os.path.split(sensor_folder)[-1]
            label = device + " " + sensor
            print(label)
            for trial_user_file in sorted(glob.glob(sensor_folder + "/*.txt")):
                df = pd.read_csv(trial_user_file, header=None)
                df = df[[0, 1, 3, 4, 5]]
                df.columns = ['user-id', 'activity', 'x-axis', 'y-axis', 'z-axis']
                df = df.astype({'user-id': 'string'})
                df['z-axis'].replace({';': ''}, regex=True, inplace=True)
                df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')
                df['device'] = label
                all_data.append(df)
                if sensor == 'accel':
                    acc_data.append(df)
                elif sensor == 'gyro':
                    gyro_data.append(df)
    all_df = pd.concat(all_data)
    acc_df = pd.concat(acc_data)
    gyro_df = pd.concat(gyro_data)

    all_users = all_df['user-id'].unique()
    acc_dataset = {}
    for user in all_users:
        user_extract = acc_df[acc_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        acc_dataset[user] = [(data, labels)]

    gyro_dataset = {}
    for user in all_users:
        user_extract = gyro_df[gyro_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        gyro_dataset[user] = [(data, labels)]

    all_dataset = {}
    for user in all_users:
        user_extract = all_df[all_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        all_dataset[user] = [(data, labels)]

    user_dataset = {
        'acc': acc_dataset,
        'gyro': gyro_dataset,
        'all': all_dataset
    }
    return user_dataset


def process_WISDM_all_har_data_standardised(data_folder_path):
    map = {
        'A': 'walking',
        'B': 'jogging',
        'C': 'stairs',
        'D': 'sitting',
        'E': 'standing',
        'F': 'typing',
        'G': 'teeth',
        'H': 'soup',
        'I': 'chips',
        'J': 'pasta',
        'K': 'drinking',
        'L': 'sandwich',
        'M': 'kicking',
        'O': 'catch',
        'P': 'dribbling',
        'Q': 'writing',
        'R': 'clapping',
        'S': 'folding'
    }

    all_device_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = []
    acc_data = []
    gyro_data = []
    for folder in all_device_folders:
        device = os.path.split(folder)[-1]
        all_sensor_folders = sorted(glob.glob(folder + "/*"))
        for sensor_folder in all_sensor_folders:
            sensor = os.path.split(sensor_folder)[-1]
            label = device + " " + sensor
            print(label)
            for trial_user_file in sorted(glob.glob(sensor_folder + "/*.txt")):
                df = pd.read_csv(trial_user_file, header=None)
                df[1].replace(map, inplace=True)
                df = df[[0, 1, 3, 4, 5]]
                df.columns = ['user-id', 'activity', 'x-axis', 'y-axis', 'z-axis']
                df['z-axis'].replace({';': ''}, regex=True, inplace=True)
                df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')
                df = df.astype({'user-id': 'string'})
                df['device'] = label
                all_data.append(df)
                if sensor == 'accel':
                    acc_data.append(df)
                elif sensor == 'gyro':
                    gyro_data.append(df)
    all_df = pd.concat(all_data)
    acc_df = pd.concat(acc_data)
    gyro_df = pd.concat(gyro_data)
    all_df = normalize_sampling_rate(all_df, '50L')
    acc_df = normalize_sampling_rate(acc_df, '50L')
    gyro_df = normalize_sampling_rate(gyro_df, '50L')

    all_users = all_df['user-id'].unique()
    acc_dataset = {}
    for user in all_users:
        user_extract = acc_df[acc_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        acc_dataset[user] = [(data, labels)]

    gyro_dataset = {}
    for user in all_users:
        user_extract = gyro_df[gyro_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        gyro_dataset[user] = [(data, labels)]

    all_dataset = {}
    for user in all_users:
        user_extract = all_df[all_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["activity"]
        all_dataset[user] = [(data, labels)]

    user_dataset = {
        'acc': acc_dataset,
        'gyro': gyro_dataset,
        'all': all_dataset
    }
    return user_dataset


def process_WISDM_all_data_standardised(data_folder_path):
    all_device_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = []
    acc_data = []
    gyro_data = []
    for folder in all_device_folders:
        device = os.path.split(folder)[-1]
        all_sensor_folders = sorted(glob.glob(folder + "/*"))
        for sensor_folder in all_sensor_folders:
            sensor = os.path.split(sensor_folder)[-1]
            label = device + " " + sensor
            print(label)
            for trial_user_file in sorted(glob.glob(sensor_folder + "/*.txt")):
                df = pd.read_csv(trial_user_file, header=None)
                df = df[[0, 1, 3, 4, 5]]
                df.columns = ['user-id', 'activity', 'x-axis', 'y-axis', 'z-axis']
                df = df.astype({'user-id': 'string'})
                df['z-axis'].replace({';': ''}, regex=True, inplace=True)
                df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')
                df['device'] = label
                all_data.append(df)
                if sensor == 'accel':
                    acc_data.append(df)
                elif sensor == 'gyro':
                    gyro_data.append(df)
    all_df = pd.concat(all_data)
    acc_df = pd.concat(acc_data)
    gyro_df = pd.concat(gyro_data)
    all_df = normalize_sampling_rate(all_df, '50L')
    acc_df = normalize_sampling_rate(acc_df, '50L')
    gyro_df = normalize_sampling_rate(gyro_df, '50L')

    all_users = all_df['user-id'].unique()
    acc_dataset = {}
    for user in all_users:
        user_extract = acc_df[acc_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        acc_dataset[user] = [(data, labels)]

    gyro_dataset = {}
    for user in all_users:
        user_extract = gyro_df[gyro_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        gyro_dataset[user] = [(data, labels)]

    all_dataset = {}
    for user in all_users:
        user_extract = all_df[all_df["user-id"] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract["device"]
        all_dataset[user] = [(data, labels)]

    user_dataset = {
        'acc': acc_dataset,
        'gyro': gyro_dataset,
        'all': all_dataset
    }
    return user_dataset


def process_WISDM_v1_HAR(data_folder_path):
    """
    This is the WISDM that SelfHAR uses
    """
    columns = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    print("here")
    df = pd.read_csv(os.path.join(data_folder_path, 'WISDM_ar_v1.1_raw.txt'), header=None, names=columns,
                     on_bad_lines='skip')
    df['z-axis'] = df['z-axis'].map(lambda x: str(re.findall(r"\d+\.\d+", str(x))))
    df['z-axis'] = df['z-axis'].map(lambda x: x[2:-2])
    df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')
    df = df.dropna()

    all_users = df['user-id'].unique()
    acc_dataset = {}
    for user in all_users:
        user_extract = df[df['user-id'] == user]
        data = user_extract[['x-axis', 'y-axis', 'z-axis']]
        labels = user_extract['activity']
        acc_dataset[user] = [(data, labels)]
    # print(f"wisdm dataset: {acc_dataset}")
    return {
        'acc': acc_dataset,
        'all': acc_dataset
    }


def store_pickle(dataset, filename):
    with open(filename + '.pickle', 'wb') as file:
        pickle.dump(dataset, file)


def open_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def normalize_sampling_rate(data, oroginal_sr, target_sr='20L'):
    df = data.copy(deep=True)
    df.reset_index(inplace=True, drop=True)

    # Create a dummy timestamp column
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq=oroginal_sr)

    # Set the 'user-id', 'activity', and 'timestamp' columns as the index
    df.set_index('timestamp', inplace=True)

    resampled_df = df.resample(target_sr).asfreq()
    resampled_df['x-axis'] = resampled_df['x-axis'].interpolate('linear', limit=2)
    resampled_df['y-axis'] = resampled_df['y-axis'].interpolate('linear')
    resampled_df['z-axis'] = resampled_df['z-axis'].interpolate('linear')
    resampled_df.reset_index(inplace=True, drop=True)
    resampled_df = resampled_df.ffill()

    return resampled_df
