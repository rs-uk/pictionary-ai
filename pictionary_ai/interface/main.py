from pictionary_ai.params import *
from pictionary_ai.utils import *
from pictionary_ai.main.preprocessor import *
from google.cloud import storage
from tqdm.auto import tqdm
from pathlib import Path
import random


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


def select_subset(dataset_local_path:str = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}", pc_within_class:int = 10, **kwargs):
    '''
    Select a subset of the locally-stored quickdraw dataset, pulling the given percentage
    of drawings within each class. This shuffles the classes and the drawings within.
    By default we take 10% of the classes as this seems enough for learning.
    **kwargs:
        - nb_classes: int, the number of classes to select at random
        - list_classes: list, the name of the classes to select, overrides nb_classes
    '''
    # List the classes present in the dataset_local_path
    list_classes = []
    classes_files = [file for file in os.listdir(dataset_local_path) if os.path.isfile(os.path.join(dataset_local_path, file)) and file.endswith('.ndjson')]
    for class_file in classes_files:
        # remove the file extension
        class_name = re.sub(r'\.[^.]*$', '', class_file)
        # remove all descriptive prefixes with the class name starting after the last '_'
        class_name = re.sub(r'^.*_(.+)$', r'\1', class_name)
        list_classes.append(class_name)

    # If the call specifies a list of classes, we use it as is
    if kwargs['list_classes'] is not None and isinstance(kwargs['list_classes'], list):
        list_classes = kwargs['list_classes']
    # If the call doesnt specify a list of classes but specifies a number of classes
    # to use then we take those classes randomly among the ones present in the dataset
    elif kwargs['nb_classes'] is not None and isinstance(kwargs['nb_classes'], int):
        list_classes = random.shuffle(list_classes)
        list_classes = list_classes[0: kwargs['nb_classes']]

    # Defining the new folder for that subset and create it if not existent
    nb_classes = len(list_classes)
    subset_local_path = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{pc_within_class}pc_{nb_classes}classes"
    if not os.path.exists(subset_local_path):
        os.makedirs(subset_local_path)

    # Build the tqdm bar
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_class_names = tqdm(list_classes, bar_format=bar_format) # to tqdm to display progress

    # Copying the selected classes files to the new subset folder
    for class_name in tqdm_class_names:
        list_drawings = []
        # Looking for the class file using the class name
        class_filename = re.search(class_name, classes_files)
        class_filepath = f"{dataset_local_path}/{class_filename}"
        # Counting the drawings in the class and computing the number to extract
        nb_drawings_in_class = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', class_filepath]))).group())
        nb_drawings_to_load = int(nb_drawings_in_class * pc_within_class / 100)
        # Build a list of the random line numbers to use within that class
        drawings_to_load = random.sample(range(nb_drawings_in_class), nb_drawings_to_load)
        # Copying the randomly selected drawings to a new class file
        for i in drawings_to_load:
            json_drawing = ujson.loads(linecache.getline(class_filepath, i+1 , module_globals=None))
            list_drawings.append(json_drawing)
        linecache.clearcache()
        tqdm_class_names.set_description(f"Extracting {pc_within_class} percent of {class_name}")
        save_drawings_to_ndjson_local(list_drawings, output_file=f"{subset_local_path}/{pc_within_class}pc_{class_filename}")
