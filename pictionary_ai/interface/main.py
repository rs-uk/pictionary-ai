from pictionary_ai.params import *
from pictionary_ai.utils import *
from pictionary_ai.main.preprocessor import *
from pictionary_ai.model.models import *
from google.cloud import storage
from tqdm.auto import tqdm
from pathlib import Path
import random
import os
from sklearn.model_selection import train_test_split



def download_simplified_dataset(source_bucket:str = BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                                prefix_blobs_source:str = None,
                                destination_path:str = LOCAL_DATA_PATH
                                ) -> None:
    '''
    Download the dataset on the machine for faster training.
    '''
    # Checking that the project's bucket matches the Google original one, if not copy Google data
    bucket_ready, reason = compare_buckets(BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                                           ORIGINAL_BUCKET_DRAWINGS,
                                           prefix_blobs1=prefix_blobs_source,
                                           prefix_blobs2=ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX
                                           )
    if not bucket_ready:
        print(reason)
        copy_bucket(ORIGINAL_BUCKET_DRAWINGS,
                    BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                    prefix_blobs_source=ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX
                    )
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(source_bucket)
    # List the blobs in the bucket (to tqdm to display progress)
    tqdm_blobs = tqdm(bucket.list_blobs(prefix=prefix_blobs_source))
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{destination_path}/{source_bucket}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Download all blobs to the destination folder
    for blob in tqdm_blobs:
        destination_path = f"{destination_folder_path}/{blob.name}"
        blob.download_to_filename(destination_path)
    print(f"Downloaded the bucket {source_bucket} locally")


def preprocess_simplified_dataset(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                                  shuffle:bool = True
                                  ) -> None:
    '''
    Process the locally-stored simplified dataset and save the created NDJSON files
    into a separate folder.
    '''
    # List the class files in the local dataset (each class is in one NDJSON file)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = LOCAL_DRAWINGS_SIMPLIFIED_PREPROCESSED_PATH
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"Processing {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_class_processed = preprocess_class(class_filepath, 'all', shuffle=shuffle)
        save_drawings_to_ndjson_local(list_class_processed, destination_folder_path)


def pad_preprocessed_dataset(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PREPROCESSED_PATH) -> None:
    '''
    Pad the locally-stored processed dataset and save the created NDJSON files
    into a separate folder.
    '''
    # List the class files in the local dataset (each class is in one NDJSON file of preprocessed drawings)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = LOCAL_DRAWINGS_SIMPLIFIED_PADDED_PATH
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"Padding {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_drawings = list_drawings_in_class(class_filepath)
        list_class_padded = pad_class(list_drawings)
        save_drawings_to_ndjson_local(list_class_padded, destination_folder_path)


def OHE_padded_dataset(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PADDED_PATH) -> None:
    '''
    OHE the locally-stored padded dataset and save the created NDJSON files
    into a separate folder.
    '''
    dict_OHE_mapping = create_classes_mapping(dataset_local_path)
    # List the class files in the local dataset (each class is in one NDJSON file of preprocessed drawings)
    class_files = [file.name for file in Path(dataset_local_path).iterdir() if file.is_file()]
    tqdm_class_files = tqdm(class_files) # to tqdm to display progress
    # Define the destination folder and create it if not existent
    destination_folder_path = LOCAL_DRAWINGS_SIMPLIFIED_OHE_PATH
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Process and save all the class files
    for class_file in tqdm_class_files:
        tqdm_class_files.set_description(f"One-Hot-Encoding {class_file}")
        class_filepath = f"{dataset_local_path}/{class_file}"
        list_drawings = list_drawings_in_class(class_filepath)
        list_class_OHE = OHE_class(list_drawings, dict_OHE_mapping)
        save_drawings_to_ndjson_local(list_class_OHE, destination_folder_path)


def preprocess_pad_OHE_simplified_dataset(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                                          shuffle:bool = True
                                          ) -> None:
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
    destination_folder_path = LOCAL_DRAWINGS_SIMPLIFIED_PROCESSED_PATH
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


def generate_subset_Xy(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                       pc_within_class:int = PERCENT_CLASS,
                       nb_classes:int = NUMBER_CLASSES,
                       list_classes:str = None
                       ) -> list:
    '''
    Select a subset of the locally-stored quickdraw dataset, pulling the given percentage
    of drawings within each class. This shuffles the classes and the drawings within, and
    saves them in a new folder describing the subset. We then collate all the classes in a
    list, shuffle that list and ready it for training. We save the collated Xy data as a
    JSON file in a new folder.
    By default we take 10% of the classes as this seems enough for learning.
    **kwargs:
        - nb_classes: int, the number of classes to select at random
        - list_classes: list, the name of the classes to select, overrides nb_classes
    Return a shuffled list of all the drawings in the subset, as dict with key-value pairs:
        - key_id: str, the UID of the drawing
        - class: str, the name of the class
        - length: int, the lenght of the drawing (nb of points before padding)
        - list_deltas: list, the drawing represented by its deltas
        - OHE_class: list, the OHE of the drawing
    '''
    # List the classes present in the dataset_local_path
    # The dict keys are the class names and the values are the file names
    dict_classes = {}
    classes_files = [file for file in os.listdir(dataset_local_path) if os.path.isfile(os.path.join(dataset_local_path, file)) and file.endswith('.ndjson')]
    for class_file in classes_files:
        # remove the file extension
        class_name = re.sub(r'\.[^.]*$', '', class_file)
        # remove all descriptive prefixes with the class name starting after the last '_'
        class_name = re.sub(r'^.*_(.+)$', r'\1', class_name)
        dict_classes[class_name] = class_file

    # If the call specifies a list of classes, we use it as is
    if list_classes is not None and isinstance(list_classes, list):
        list_classes = list_classes
    # If the call doesnt specify a list of classes but specifies a number of classes
    # to use then we take those classes randomly among the ones present in the dataset
    elif nb_classes is not None and isinstance(nb_classes, int):
        list_classes = list(dict_classes.keys())
        random.shuffle(list_classes)
        list_classes = list_classes[:nb_classes]

    # Defining the new folder for that subset and create it if not existent
    nb_classes = len(list_classes)
    subset_local_path = LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH
    if not os.path.exists(subset_local_path):
        os.makedirs(subset_local_path)

    # Build the tqdm bar
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_class_names = tqdm(list_classes, bar_format=bar_format) # to tqdm to display progress

    # Copying the selected classes files to the new subset folder and storing all drawings (dictionaries) in a list
    list_subset_drawings = []
    for class_name in tqdm_class_names:
        list_class_drawings = []
        # Getting the class file from the class name
        class_filepath = f"{dataset_local_path}/{dict_classes[class_name]}"
        # Counting the drawings in the class and computing the number to extract
        nb_drawings_in_class = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', class_filepath]))).group())
        nb_drawings_to_load = int(nb_drawings_in_class * pc_within_class / 100)
        # Build a list of the random line numbers to use within that class (the seed helps to freeze that for analysis)
        random.seed(42)
        drawings_to_load = random.sample(range(nb_drawings_in_class), nb_drawings_to_load)
        # Copying the randomly selected drawings to a new class file
        for i in drawings_to_load:
            json_drawing = ujson.loads(linecache.getline(class_filepath, i+1 , module_globals=None))
            list_class_drawings.append(json_drawing)
        linecache.clearcache()
        tqdm_class_names.set_description(f"Extracting {pc_within_class} percent of {class_name}")
        save_drawings_to_ndjson_local(list_class_drawings, output_file=f"{subset_local_path}/{pc_within_class}pc_{dict_classes[class_name]}")
        # We concatenate the class drawings to the subset drawings' list (NOT APPENDING)
        list_subset_drawings = list_subset_drawings + list_class_drawings

    preprocess_pad_OHE_simplified_dataset(dataset_local_path=subset_local_path)

    # We shuffle the drawings in the subset
    random.shuffle(list_subset_drawings)

    return list_subset_drawings


def split_Xy(list_subset_drawings:list) -> dict:
    '''
    Split the Xy collated data into X_train, X_val, y_train, y_val, X_test, y_test.
    We assume the drawings are already shuffled (done when generating the subset)
    Return a dictionary with keys:
        - X_train
        - X_val
        - y_train
        - y_val
        - X_test
        - y_test
    Each values are lists of lists for X and lists for y.
    '''
    X = [drawing['list_deltas'] for drawing in list_subset_drawings]
    y = [drawing['OHE_class'] for drawing in list_subset_drawings]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

    dict_split_dataset = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}

    # Save the dictionary in case needed
    destination_folder_path = LOCAL_DRAWINGS_SIMPLIFIED_MODELREADY_PATH
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Build the tqdm bar
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_Xy = tqdm(dict_split_dataset.items(), bar_format=bar_format) # to tqdm to display progress

    for key, value in tqdm_Xy:
        output_filename = key + '.json'
        output_filepath = f"{destination_folder_path}/{output_filename}"
        tqdm_Xy.set_description(f"Saving {output_filename}")
        with open(output_filepath, 'w') as json_file:
            dict_variable = {key: value}
            ujson.dump(dict_variable, json_file)

    return dict_split_dataset


def train_model(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                pc_within_class:int = PERCENT_CLASS,
                nb_classes:int = NUMBER_CLASSES
                ) -> Tuple[Model, dict]:
    '''
    Initialize, compile and train the model.
    '''
    model = initialize_model()
    model = compile_model(model, learning_rate=0.0005)
    list_subset_drawings = generate_subset_Xy(dataset_local_path, pc_within_class=pc_within_class, nb_classes=nb_classes)
    dict_Xy = split_Xy(list_subset_drawings)
    model, history = train_model(model,
                                 X = np.ndarray(dict_Xy['X_train']),
                                 y = np.ndarray(dict_Xy['y_train']),
                                 batch_size=256,
                                 patience=3,
                                 validation_data=[np.ndarray(dict_Xy['X_val']), np.ndarray(dict_Xy['y_val'])]
                                 )
    return model, history
