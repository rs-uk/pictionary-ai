from pictionary_ai.params import *
from pictionary_ai.utils import *
from pictionary_ai.main.preprocessor import *
from google.cloud import storage
from tqdm.auto import tqdm
from pathlib import Path


def download_simplified_dataset(source_bucket:str = BUCKET_NAME_DRAWINGS_SIMPLIFIED, destination_path:str = LOCAL_DATA_PATH) -> None:
    '''
    Download the dataset on the machine for faster training.
    '''
    # Checking that the project's bucket matches the Google original one, if not copy Google data
    bucket_ready, reason = compare_buckets(BUCKET_NAME_DRAWINGS_SIMPLIFIED, ORIGINAL_BUCKET_DRAWINGS_SIMPLIFIED)
    if not bucket_ready:
        print(reason)
        copy_bucket(ORIGINAL_BUCKET_DRAWINGS_SIMPLIFIED, BUCKET_NAME_DRAWINGS_SIMPLIFIED)
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(source_bucket)
    # List the blobs in the bucket (to tqdm to display progress)
    tqdm_blobs = tqdm(bucket.list_blobs())
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{destination_path}/{source_bucket}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Download all blobs to the destination folder
    for blob in tqdm_blobs:
        destination_path = f"{destination_folder_path}/{blob.name}"
        blob.download_to_filename(destination_path)
    print(f"Downloaded the bucket {source_bucket} locally")


def preprocess_simplified_dataset(dataset_local_path:str = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}", shuffle:bool = True) -> None:
    '''
    Process the locally-stored simplified dataset and save the created NDJSON files
    into a separate folder.
    '''
    # List the class files in the local dataset (each class is in one NDJSON file)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{LOCAL_DATA_PATH}/preprocessed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"Processing {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_class_processed = preprocess_class(class_filepath, 'all', shuffle=shuffle)
        save_drawings_to_ndjson_local(list_class_processed, destination_folder_path)


def pad_preprocessed_dataset(dataset_local_path:str = f"{LOCAL_DATA_PATH}/preprocessed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}") -> None:
    '''
    Pad the locally-stored processed dataset and save the created NDJSON files
    into a separate folder.
    '''
    # List the class files in the local dataset (each class is in one NDJSON file of preprocessed drawings)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{LOCAL_DATA_PATH}/padded_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"Padding {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_drawings = list_drawings_in_class(class_filepath)
        list_class_padded = pad_class(list_drawings)
        save_drawings_to_ndjson_local(list_class_padded, destination_folder_path)


def OHE_padded_dataset(dataset_local_path:str = f"{LOCAL_DATA_PATH}/padded_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}") -> None:
    '''
    OHE the locally-stored padded dataset and save the created NDJSON files
    into a separate folder.
    '''
    dict_OHE_mapping = create_classes_mapping(dataset_local_path)
    # List the class files in the local dataset (each class is in one NDJSON file of preprocessed drawings)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{LOCAL_DATA_PATH}/OHE_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"One-Hot-Encoding {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_drawings = list_drawings_in_class(class_filepath)
        list_class_OHE = OHE_class(list_drawings, dict_OHE_mapping)
        save_drawings_to_ndjson_local(list_class_OHE, destination_folder_path)


def preprocess_pad_OHE_simplified_dataset(dataset_local_path:str = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}", shuffle:bool = True) -> None:
    '''
    Process the locally-stored simplified dataset and save the created NDJSON files
    into a separate folder. The output is model-ready files for each class.
    '''
    # Create the classes mapping dictionary
    dict_OHE_mapping = create_classes_mapping(dataset_local_path)
    # List the class files in the local dataset (each class is in one NDJSON file)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    # Build the tqdm bar
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_class_files = tqdm(class_files, bar_format=bar_format) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{LOCAL_DATA_PATH}/fully-processed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Process and save all the class files
    for class_file in tqdm_class_files:
        class_filepath = f"{dataset_local_path}/{class_file}"
        tqdm_class_files.set_description(f"Pre-processing {class_file}")
        list_drawings_processed = preprocess_class(class_filepath, 'all', shuffle=shuffle, silent=False)
        tqdm_class_files.set_description(f"Padding {class_file}")
        list_drawings_padded = pad_class(list_drawings_processed, silent=False)
        tqdm_class_files.set_description(f"One-Hot-Encoding {class_file}")
        list_drawings_OHE = OHE_class(list_drawings_padded, dict_OHE_mapping, silent=False)
        tqdm_class_files.set_description(f"Saving {class_file}")
        save_drawings_to_ndjson_local(list_drawings_OHE, f"{destination_folder_path}/{class_file}", silent=False)


# TODO: input a number of classes and % within the classes to use

def select_subset(nb_classes)
