import os
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm

DATASET_METADATA = {
    'motionsense': {
        'name': 'motionsense',
        'dataset_home_page': 'https://github.com/mmalekzadeh/motion-sense/',
        'source_url': 'https://github.com/mmalekzadeh/motion-sense/blob/master/data/C_Gyroscope_data.zip?raw=true',
        'file_name': 'C_Gyroscope_data.zip',
    },
    'hhar': {
        'name': 'hhar',
        'dataset_home_page': 'http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition',
        'source_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip',
        'file_name': 'Activity recognition exp.zip',
    },
    'wisdm': {
        'name': 'wisdm',
        'dataset_home_page': 'https://www.cis.fordham.edu/wisdm/dataset.php',
        'source_url': 'https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz',
        'file_name': 'WISDM_ar_latest.tar.gz',
    },
    'dailysports': {
        'name': 'dailysports',
        'dataset_home_page': 'https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities',
        'source_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip',
        'file_name': 'data.zip',
    },
    'harth': {
        'name': 'harth',
        'dataset_home_page': 'https://archive.ics.uci.edu/dataset/779/harth',
        'source_url': 'https://archive.ics.uci.edu/static/public/779/harth.zip',
        'file_name': 'HARTH.zip',
    },
    'pamap2': {
        'name': 'pamap2',
        'dataset_home_page': 'http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring',
        'source_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip',
        'file_name': 'PAMAP2_Dataset.zip',
    }
}

ORIGINAL_DATASET_SUB_DIRECTORY = 'original_datasets'


def get_parser():
    parser = argparse.ArgumentParser(
        description='Download and unzip HAR datasets')
    parser.add_argument('--working_directory', default='test_run',
                        help='the output directory of the downloads and processed datasets')
    parser.add_argument('--dataset', default='all',
                        choices=['motionsense', 'hhar', 'wisdm', 'dailysports', 'harth', 'pamap2', 'all'],
                        help='name of the dataset to be downloaded')
    return parser


def download_dataset(data_directory, dataset_metadata):
    dataset_name = dataset_metadata['name']
    dataset_url = dataset_metadata['source_url']
    file_name = dataset_metadata['file_name']

    if not os.path.exists(os.path.join(data_directory, dataset_name)):
        os.makedirs(os.path.join(data_directory, dataset_name))

    print(f"Downloading {dataset_name}...")

    # Stream the download and show a progress bar
    response = requests.get(dataset_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(os.path.join(data_directory, dataset_name, file_name), 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print(f"ERROR: Something went wrong with the download of {dataset_name}")

    print(f"Finished downloading to {os.path.join(data_directory, dataset_name, file_name)}")

    print(f"Unzipping {file_name}...")
    if file_name.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(data_directory, dataset_name, file_name), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_directory, dataset_name))
    elif file_name.endswith('.tar') or file_name.endswith('.tar.gz'):
        with tarfile.open(os.path.join(data_directory, dataset_name, file_name), 'r') as tar_ref:
            tar_ref.extractall(os.path.join(data_directory, dataset_name))
    else:
        print(f"Unsupported file format: {file_name}")
        return

    print(f"Finished unzipping {file_name}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.working_directory):
        os.makedirs(args.working_directory)
    dataset_directory = os.path.join(args.working_directory, ORIGINAL_DATASET_SUB_DIRECTORY)
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    if args.dataset == 'all':
        datasets = list(DATASET_METADATA.keys())
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(f"-------- Downloading {dataset} --------")
        download_dataset(dataset_directory, DATASET_METADATA[dataset])
    print("Finished")
