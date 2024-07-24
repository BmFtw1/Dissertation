'''
This file is used to pre-process all of the raw data and turn them into pickled files to be used. 
This should only be run once
'''

from raw_data_processing import process_hhar_all_files, process_hhar_all_har_files, process_motion_sense_all_files, process_motion_sense_all_har_files, process_PAMAP2_all_data, process_pamap2_har_files, process_HARTH_all_har_data, process_HARTH_all_acc_data, process_dasa_all_har_data, process_dasa_all_data, process_WISDM_all_data, process_WISDM_all_har_data
import pickle
import raw_data_processing

if __name__ == '__main__':
    motion_sense_df = process_motion_sense_all_files("test_run/original_datasets/motionsense")
    pamap_df = process_PAMAP2_all_data("test_run/original_datasets/PAMAP2/PAMAP2_Dataset/Protocol")
    dasa_df = process_dasa_all_data("test_run/original_datasets/dailysports/data")
    hhar_df = process_hhar_all_files("test_run/original_datasets/hhar/Activity recognition exp")
    harth_df = process_HARTH_all_acc_data("test_run/original_datasets/harth/harth")
    #wisdm_df = process_WISDM_all_data("test_run/original_datasets/wisdm-dataset/raw")


    # pickle the datasets
    with open('pickled_datasets/pamap2.pickle', 'wb') as file:
        pickle.dump(pamap_df, file)
    with open('pickled_datasets/hhar2.pickle', 'wb') as file:
        pickle.dump(hhar_df, file)
    with open('pickled_datasets/motionsense2.pickle', 'wb') as file:
        pickle.dump(motion_sense_df, file)
    with open('pickled_datasets/harth2.pickle', 'wb') as file:
        pickle.dump(harth_df, file)
    with open('pickled_datasets/dasa2.pickle', 'wb') as file:
        pickle.dump(dasa_df, file)


    motion_sense_har_df = process_motion_sense_all_har_files("test_run/original_datasets/motionsense")
    hhar_har_df = process_hhar_all_har_files("test_run/original_datasets/hhar/Activity recognition exp")
    dasa_har_df = process_dasa_all_har_data("test_run/original_datasets/dailysports/data")
    pamap_har_df = process_pamap2_har_files("test_run/original_datasets/PAMAP2/PAMAP2_Dataset/Protocol")
    harth_har_df = process_HARTH_all_har_data("test_run/original_datasets/harth/harth")
    # wisdm_har_df = process_WISDM_all_har_data("test_run/original_datasets/wisdm-dataset/raw")
    wisdm1_har_df = raw_data_processing.process_WISDM_v1_HAR("test_run/original_datasets/wisdm/WISDM_ar_v1.1")

    with open('pickled_datasets/pamap_har.pickle', 'wb') as file:
        pickle.dump(pamap_har_df, file)
    with open('pickled_datasets/hhar_har.pickle', 'wb') as file:
        pickle.dump(hhar_har_df, file)
    with open('pickled_datasets/motionsense_har.pickle', 'wb') as file:
        pickle.dump(motion_sense_har_df, file)
    with open('pickled_datasets/harth_har.pickle', 'wb') as file:
        pickle.dump(harth_har_df, file)
    with open('pickled_datasets/dasa_har.pickle', 'wb') as file:
        pickle.dump(dasa_har_df, file)
    # with open('pickled_datasets/wisdm_har.pickle', 'wb') as file:
    #     pickle.dump(wisdm_har_df, file)
    with open('pickled_datasets/wisdm1_har.pickle', 'wb') as file:
        pickle.dump(wisdm1_har_df, file) 