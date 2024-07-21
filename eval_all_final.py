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
import pandas as pd
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
with open('pickled_datasets/wisdm2.pickle', 'rb') as file:
    wisdm_df = pickle.load(file)

    
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
with open('pickled_datasets/wisdm_har.pickle', 'rb') as file:
    wisdm_har_df = pickle.load(file)

datasets = [
    ("HHAR", hhar_df, hhar_har_df, 'acc', 'acc', {}),
    ("MotionSense", motion_sense_df, motionsense_har_df, 'all', 'acc', {}),
    ("PAMAP", pamap_df, pamap_har_df, 'acc', 'acc', {}),
    ("HARTH", harth_df, harth_har_df, 'acc', 'acc', {}),
    ("DASD", dasa_df, dasa_har_df, 'acc', 'acc', {'shift': 10}),
    ("WISDM", wisdm_df, wisdm_har_df, 'acc', 'acc', {'step': 2})
]

results = []

for dataset_name, dataset, har_dataset, target_columns, label_column, additional_params in datasets:
    result = Evaluation1.eval_downstream_model(dataset, har_dataset, target_columns, label_column, core_model="CNN", **additional_params)
    results.append((dataset_name, result))

# Print results
for dataset_name, result in results:
    print(dataset_name)
    print(result)
    print()

# print("HHAR")
# print(Evaluation1.eval_downstream_model(hhar_df, hhar_har_df, 'acc', 'acc'))

# print("MotionSense")
# print(Evaluation1.eval_downstream_model(motion_sense_df, motionsense_har_df, 'all', 'acc'))

# print("PAMAP")
# print(Evaluation1.eval_downstream_model(pamap_df, pamap_har_df, 'acc', 'acc'))

# print("HARTH")
# print(Evaluation1.eval_downstream_model(harth_df, harth_har_df, 'acc', 'acc'))

# print("DASD")
# print(Evaluation1.eval_downstream_model(dasa_df, dasa_har_df, 'acc', 'acc', shift=10))

# print("WISDM")
# print(Evaluation1.eval_downstream_model(wisdm_df, wisdm_har_df, 'acc', 'acc', step=2))