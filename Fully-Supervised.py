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
with open('pickled_datasets/wisdm1_har.pickle', 'rb') as file:
    wisdm1_har_df = pickle.load(file)

print(Evaluation1.eval_fully_supervised(pamap_har_df))
print("evaluated pamap")

print(Evaluation1.eval_fully_supervised(hhar_har_df))
print("evaluated hhar_har_df")

print(Evaluation1.eval_fully_supervised(motionsense_har_df))
print("evaluated motionsense_har_df")

print(Evaluation1.eval_fully_supervised(harth_har_df))
print("evaluated harth_har_df")

print(Evaluation1.eval_fully_supervised(dasa_har_df, 10))
print("evaluated dasa_har_df")

# print(Evaluation1.eval_fully_supervised(wisdm_har_df))
# print("evaluated wisdm_har_df")

print(Evaluation1.eval_fully_supervised(wisdm1_har_df, step=2))
print("evaluated wisdm1_har_df")
