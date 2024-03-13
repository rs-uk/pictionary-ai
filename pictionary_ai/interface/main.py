from pictionary_ai.params import *
from pictionary_ai.utils import *
from pictionary_ai.main.preprocessor import *
from pictionary_ai.model.models import *
from google.cloud import storage
from tqdm.auto import tqdm
from pathlib import Path
import random
import os, re
from sklearn.model_selection import train_test_split
from datetime import datetime


# TODO: check that data to download is not already stored locally

def download_simplified_dataset(source_bucket:str = BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                                source_folder_path:str = None,
                                destination_path:str = LOCAL_RAW_DATA_PATH
                                ) -> None:
    '''
    Download the dataset on the machine for faster training.
    '''
    # Checking that the project's bucket matches the Google original one, if not copy Google data
    bucket_ready, reason = compare_buckets(BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                                           ORIGINAL_BUCKET_DRAWINGS,
                                           folder1_path=source_folder_path,
                                           folder2_path=ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX
                                           )
    if not bucket_ready:
        copy_bucket(ORIGINAL_BUCKET_DRAWINGS,
                    BUCKET_NAME_DRAWINGS_SIMPLIFIED,
                    folder1_path=ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX
                    )
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(source_bucket)
    # List the blobs in the bucket (to tqdm to display progress)
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_blobs = tqdm(bucket.list_blobs(prefix=source_folder_path),
                      bar_format=bar_format,
                      total=len(list(bucket.list_blobs(prefix=source_folder_path))))
    # Define the destination folder and create it if not existent
    destination_folder_path = f"{destination_path}/{source_bucket}"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # Download all blobs to the destination folder
    for blob in tqdm_blobs:
        tqdm_blobs.set_description(f"Downloading {blob.name}")
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


def process_dataset(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                    dataset_local_processed:str = None,
                    shuffle:bool = True,
                    save_processed_classes:bool = True,
                    ) -> dict:
    '''
    Process the locally-stored simplified dataset and save the created NDJSON files
    into a separate folder. The output is model-ready files for each class.
    Return a dictionary with key-value pairs:
    - dict_OHE: dict, each key is a class name and each value is the associated index
    in the OHE space
    - list_drawings: list, a shuffled list of all the drawings in the subset, as
    dictionaries with key-value pairs:
        - key_id: str, the UID of the drawing
        - class: str, the name of the class
        - length: int, the lenght of the drawing (nb of points before padding)
        - list_deltas: list, the drawing represented by its deltas
        - OHE_class: list, the OHE of the drawing
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
    # Define the saving folder and create it if not existent
    if save_processed_classes:
        destination_folder_path = dataset_local_processed
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)
    # Define the list to return
    list_subset_processed_drawings = []
    # Process and save all the class files
    for class_file in tqdm_class_files:
        class_filepath = f"{dataset_local_path}/{class_file}"
        tqdm_class_files.set_description(f"Pre-processing {class_file}")
        list_drawings_processed = preprocess_class(class_filepath, 'all', shuffle=shuffle, silent=False)
        tqdm_class_files.set_description(f"Padding {class_file}")
        list_drawings_padded = pad_class(list_drawings_processed, silent=False)
        tqdm_class_files.set_description(f"One-Hot-Encoding {class_file}")
        list_drawings_OHE = OHE_class(list_drawings_padded, dict_OHE_mapping, silent=False)
        # Optionally we do not store the files on the local drive and just build the
        # list in memory. Default behavior is to save.
        if save_processed_classes:
            tqdm_class_files.set_description(f"Saving {class_file}")
            save_drawings_to_ndjson_local(list_drawings_OHE, f"{destination_folder_path}/{class_file}", silent=False)
        # We concatenate the class drawings to the subset drawings' list (NOT APPENDING)
        list_subset_processed_drawings += list_drawings_OHE

    output = {}
    output['dict_OHE'] = dict_OHE_mapping
    output['list_drawings'] = list_subset_processed_drawings

    return output


def generate_subset_Xy(dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                       pc_within_class:int = PERCENT_CLASS,
                       nb_classes:int = NUMBER_CLASSES,
                       list_classes:list = None, # overrides nb_classes
                       save_processed_classes:bool = True,
                       ) -> dict:
    '''
    Select a random subset of the locally-stored quickdraw dataset, pulling the given
    percentage of drawings within each class at random. The new classes are then
    stored locally in a separate folder describing the subset (number of classes and
    percentage used within each class).
    This subset is then processed (pre-processed + padded + OHE'd) and stored in another
    folder. We then collate all the classes in a list, shuffle that list and return
    it as the output.
    By default we take 10% of the classes as this seems enough for learning.
    **kwargs:
        - nb_classes: int, the number of classes to select at random
        - list_classes: list, the name of the classes to select, overrides nb_classes
    Return a dictionary with:
        - dict_OHE: dict, each key is a class name and each value is the associated index
        in the OHE space
        - list_drawings: list, a shuffled list of all the drawings in the subset, as
        dictionaries with key-value pairs:
            - key_id: str, the UID of the drawing
            - class: str, the name of the class
            - length: int, the lenght of the drawing (nb of points before padding)
            - list_deltas: list, the drawing represented by its deltas
            - OHE_class: list, the OHE of the drawing
    '''
    ##### List the classes present in the dataset_local_path
    # The dict keys are the class names and the values are the file names
    dict_classes = {}
    classes_files = [file for file in os.listdir(dataset_local_path) if os.path.isfile(os.path.join(dataset_local_path, file)) and file.endswith('.ndjson')]
    for class_file in classes_files:
        # remove the file extension
        class_name = re.sub(r'\.[^.]*$', '', class_file)
        # remove all descriptive prefixes with the class name starting after the last '_'
        class_name = re.sub(r'^.*_(.+)$', r'\1', class_name)
        dict_classes[class_name] = class_file

    ##### Build the list of classes to use for the subset
    # If the call specifies a list of classes, we use it as is
    if list_classes is not None and isinstance(list_classes, list):
        list_classes = list_classes
    # If the call doesnt specify a list of classes but specifies a number of classes
    # to use then we take those classes randomly among the ones present in the dataset
    elif nb_classes is not None and isinstance(nb_classes, int):
        list_classes = list(dict_classes.keys())
        random.shuffle(list_classes)
        list_classes = list_classes[:nb_classes]

    ##### Defining the new folder for that subset and create it if not existent
    if not os.path.exists(LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH):
        os.makedirs(LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH)

    nb_classes = len(list_classes)

    # Build the tqdm bar
    l_bar='{desc} {percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    tqdm_class_names = tqdm(list_classes, bar_format=bar_format) # to tqdm to display progress

    ##### Extracting the required percentage of drawings from each class and storing the
    # resampled class files to the new subset folder for processing
    for class_name in tqdm_class_names:
        list_class_drawings = []
        tqdm_class_names.set_description(f"Extracting {pc_within_class} percent of {class_name}".format(class_name))
        # Getting the class file from the class name
        class_filepath = f"{dataset_local_path}/{dict_classes[class_name]}".format(class_name)
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
        save_drawings_to_ndjson_local(list_class_drawings, output_file=f"{LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH}/{pc_within_class}pc_{dict_classes[class_name]}")
        # We concatenate the class drawings to the subset drawings' list (NOT APPENDING)
        # list_subset_drawings = list_subset_drawings + list_class_drawings

    ##### Processing all the classes in the subset folder
    dict_processed_dataset = process_dataset(dataset_local_path=LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH,
                                             dataset_local_processed=LOCAL_DRAWINGS_SIMPLIFIED_PROCESSED_PATH,
                                             save_processed_classes=save_processed_classes
                                             )
    dict_OHE = dict_processed_dataset['dict_OHE']
    list_subset_processed_drawings = dict_processed_dataset['list_drawings']

    # We shuffle the drawings in the subset
    random.shuffle(list_subset_processed_drawings)

    output = {}
    output['dict_OHE'] = dict_OHE
    output['list_drawings'] = list_subset_processed_drawings

    return output


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

    X = [(drawing)['list_deltas'] for drawing in list_subset_drawings]
    y = [(drawing)['OHE_class'] for drawing in list_subset_drawings]

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
    tqdm_Xy = tqdm(dict_split_dataset.items(), bar_format=bar_format, total=6) # to tqdm to display progress

    for key, value in tqdm_Xy:
        output_filename = key + '.json'
        output_filepath = f"{destination_folder_path}/{output_filename}"
        tqdm_Xy.set_description(f"Saving {output_filename}")
        with open(output_filepath, 'w') as json_file:
            dict_variable = {key: value}
            ujson.dump(dict_variable, json_file)

    return dict_split_dataset


def load_model(checkpoint_path:str) -> dict:
    ##### change the output to a compiled h5 model
    '''
    Return a dictionary dict_model with key-value pairs:
        - model: Model, the pretrained model to use with its checkpoint, compiled
        with its weigths loaded.
        - params: dict, a dictionary of the params with key-value pairs:
            - MAX_LENGTH: int, the length used for padding
            - PADDING_VALUE: int, the value used for padding
            - dict_OHE: dict, each key is a class name and each value is the
            associated index in the OHE space
    '''
    if not os.path.exists(checkpoint_path):
        print('The model directory does not exist.')

    model_params_filepath = os.path.join(checkpoint_path, 'params_for_training.txt')
    str_MAX_LENGTH = linecache.getline(model_params_filepath, 1, module_globals=None) # MAX_LENGTH stored on first line
    str_PADDING_VALUE = linecache.getline(model_params_filepath, 2, module_globals=None) # PADDING_VALUE stored on second line
    MAX_LENGTH = int(re.search(r'=(\d+)', str_MAX_LENGTH))
    PADDING_VALUE = re.search(r'=(\d+)', str_PADDING_VALUE)
    dict_OHE = ujson.loads(linecache.getline(model_params_filepath, 3, module_globals=None)) # dict_OHE stored as json on third line

    dict_model = {}
    params = {}
    params['MAX_LENGTH'] = MAX_LENGTH
    params['PADDING_VALUE'] = PADDING_VALUE
    params['dict_OHE'] = dict_OHE

    model = initialize_model(mask_value=PADDING_VALUE, input_shape=(MAX_LENGTH, 3))
    model = compile_model(model)
    model.load_weights(checkpoint_path)

    dict_model['model'] = model
    dict_model['params'] = params

    return dict_model


def train_model_calling(dict_model:Model = None, # overrides the model to be trained (should not use for now)
                        dataset_local_path:str = LOCAL_DRAWINGS_SIMPLIFIED_PATH,
                        pc_within_class:int = PERCENT_CLASS,
                        nb_classes:int = NUMBER_CLASSES,
                        list_classes:list = LIST_CLASSES, # overrides the nb_classes and uses the list in the shared data here
                        ) -> Tuple[Model, dict]:
    '''
    Initialize, compile and train the model.
    If the dict_model is specified, this is the one we train.
    /!\ training a model with a given list_classes for comparison of models will not
    use the same drawings in each class, but the dataset is big enough to consider
    that is not a problem.
    '''
    # if dict_model is not None:
    #     # We use the compiled model with its weigths loaded
    #     model = dict_model['model']
    #     model_params = dict_model['params']
    #     MAX_LENGTH = model_params['MAX_LENGTH']
    #     PADDING_VALUE = model_params['PADDING_VALUE']
    #     dict_OHE = model_params['dict_OHE']
    # else:
    #     # We instantiate a new model with the calling params
    model = initialize_model()
    model = compile_model(model)

    dict_subset = generate_subset_Xy(dataset_local_path,
                                     pc_within_class=pc_within_class,
                                     nb_classes=nb_classes,
                                     list_classes=list_classes,
                                     save_processed_classes=False # Switch this to true to save the files
                                     )
    list_subset_drawings = dict_subset['list_drawings']
    dict_OHE = dict_subset['dict_OHE']
    list_classes = list(dict_OHE.keys())
    dict_Xy = split_Xy(list_subset_drawings)

    ##### Saving training details and model weights/checkpoints
    # Build a folder with a name including NUMBER_CLASSES and PERCENT_CLASS along with
    # the date of training to save the model checkpoints and refer to them.
    str_start_training = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    str_unique_folder = f"{str_start_training}_{nb_classes}classes_{pc_within_class}pc"
    checkpoint_dir = f"{MODELS_PATH}/{str_unique_folder}"
    checkpoint_path = f"{MODELS_PATH}/{str_unique_folder}/checkpoint.ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # In that folder, save a file with the padding length, the padding value, the
    # One Hot Endocing dictionary and the classes used for training in alphabetical
    # order for easy reference.
    list_classes.sort()
    training_params_filepath = '/'.join((checkpoint_dir, 'params_for_training.txt'))
    with open(training_params_filepath, 'w') as file_training_params:
        file_training_params.write("%s\n" % f"MAX_LENGTH={MAX_LENGTH}")
        file_training_params.write("%s\n" % f"PADDING_VALUE={PADDING_VALUE}")
        ujson.dump(dict_OHE, file_training_params)
        file_training_params.write('\n')
        for item in list_classes:
            # Write each class name to the file followed by a newline
            file_training_params.write("%s\n" % item)

    print('Compiled the following model:')
    print(model.summary())

    model, history = train_model(model,
                                 X = np.array(dict_Xy['X_train']),
                                 y = np.array(dict_Xy['y_train']),
                                 batch_size=256,
                                 patience=3,
                                 validation_data=[np.array(dict_Xy['X_val']), np.array(dict_Xy['y_val'])],
                                 checkpoint_path=checkpoint_path
                                 )
    return model, history
