from google.cloud import storage
import linecache
import numpy as np
from tqdm.auto import tqdm
import re, os, subprocess
import ujson # much faster than json lib for simple tasks


def list_bucket_contents(bucket_name:str) -> list:
    '''
    Return a dictionary of the blobs in a bucket.
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs]


def compare_buckets(bucket1_name:str,
                    bucket2_name:str,
                    folder1_path:str = None,
                    folder2_path:str = None
                    ) -> tuple:
    '''
    Compare two buckets (or folders in those buckets) and return True if they are identical and/or an explanation.
    The folder paths should be such as 'first/second/third/target_folder'.
    '''
    bucket1_contents = set(list_bucket_contents(bucket1_name))
    bucket2_contents = set(list_bucket_contents(bucket2_name))

    if folder1_path is not None:
        folder1_path = folder1_path + '/'
    else:
        folder1_path = ''
    if folder2_path is not None:
        folder2_path = folder2_path + '/'
    else:
        folder2_path = ''

    folder1_contents = {item.removeprefix(folder1_path) for item in bucket1_contents if item.startswith(folder1_path)}
    folder2_contents = {item.removeprefix(folder2_path) for item in bucket2_contents if item.startswith(folder2_path)}

    str_path1 = '/'.join((bucket1_name, folder1_path))
    str_path2 = '/'.join((bucket2_name, folder2_path))

    # Check if the objects in the folders are the same:
    if folder1_contents != folder2_contents:
        warning = f"Files in {str_path1} but not in {str_path2}:\n \
                    {folder1_contents - folder2_contents} \n \
                    Files in {str_path2} but not in {str_path1}:\n \
                    {folder2_contents - folder1_contents}"
        return False, warning

    return True, "Buckets are identical"


def copy_bucket(bucket1_name:str,
                bucket2_name:str,
                folder1_path:str = None,
                folder2_path:str = None
                ) -> None:
    '''
    Copy all blobs from bucket1_name/folder1_path to bucket2_name/folder2_path.
    '''
    # Initialize clients for source and destination buckets
    client1 = storage.Client()
    client2 = storage.Client()

    # Get the source and destination buckets
    bucket1 = client1.get_bucket(bucket1_name)
    bucket2 = client2.get_bucket(bucket2_name)

    # List blobs in the source bucket (to tqdm to show copy progress)
    blobs = tqdm(bucket1.list_blobs(prefix=folder1_path))

    # Copy each blob to the destination bucket
    for blob in blobs:
        if folder2_path is not None:
            destination_blob_name = blob.name.replace(folder1_path, folder2_path, 1)
        else:
            destination_blob_name = blob.name
        destination_blob = bucket1.copy_blob(blob, bucket2, new_name=destination_blob_name)


def copy_bucket_objects_with_prefix(source_bucket_name,
                                    destination_bucket_name,
                                    prefix) -> None:
    # Create a client object
    client = storage.Client()

    # Get the source and destination buckets
    source_bucket = client.get_bucket(source_bucket_name)
    destination_bucket = client.get_bucket(destination_bucket_name)

    # List the blobs (objects) in the source bucket
    blobs = source_bucket.list_blobs()

    # Copy each blob to the destination bucket with the specified prefix
    for blob in blobs:
        new_blob = source_bucket.copy_blob(blob, destination_bucket, new_name=f"{prefix}/{blob.name}")
        print(f'Copied {blob.name} to {new_blob.name} in {destination_bucket_name} with prefix {prefix}')


def list_blobs(bucket_name:str) -> list:
    '''
    List all the blobs in a given bucket and return their names in a list.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List the blobs in the bucket
    blobs = bucket.list_blobs()

    # Collect the names of blobs into a list
    blob_names = [blob.name for blob in blobs]

    return blob_names


def download_blob_to_local_file(bucket_name:str,
                                source_blob_name:str,
                                destination_path:str,
                                destination_file_name:str = None
                                ) -> None:
    '''
    Downloads a blob from the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Use the blob name for the local file is a filename is not provided
    if destination_file_name is None:
        destination_file_name = blob.name

    # Define the destination file path
    destination_file_path = '/'.join((destination_path, destination_file_name))

    # Download the blob to a file
    blob.download_to_filename(destination_file_path)


def upload_blob_from_local_file(source_path:str,
                                source_file_name:str,
                                bucket_name:str,
                                destination_blob_name:str = None
                                ) -> None:
    '''
    Uploads a file to the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Use the file name for the blob name if a blob name is not provided
    if destination_blob_name is None:
        destination_blob_name = source_file_name

    # Create a blob
    blob = bucket.blob(destination_blob_name)

    # Define the source file path
    source_file_path = '/'.join((source_path, source_file_name))

    # Upload the file to the blob
    blob.upload_from_filename(source_file_path)


def load_json_for_training(ndjson_filepath:object, is_X=True):
    if is_X:
        with open(ndjson_filepath, 'r') as f:
            feature = ujson.load(f)['X_data']
        return feature
    else:
        with open(ndjson_filepath, 'r') as f:
            feature = ujson.load(f)['y_data']
        return feature


def list_drawings_in_class(class_filepath:str) -> list:
    '''
    Create a list of all drawings in a class as dict.
    '''
    list_drawings = []
    # Getting the number of lines in the file using a shell command (fastest way)
    nb_drawings = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', class_filepath]))).group())
    for i in nb_drawings:
        list_drawings.append(linecache.getline(class_filepath, i+1 , module_globals=None))

    return list_drawings


def save_drawings_to_ndjson_local(list_drawings:list, output_file:str, silent:bool = False) -> None:
    '''
    Saves the drawings in the list to a local NDJSON file.
        - list_drawings: contains a dictionary for each drawing
        - output_file: the complete filepath to the target file to save/create (.ndjson)
    '''
    l_bar='{percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt}'
    bar_format = l_bar + bar + r_bar
    tqdm_list_drawings = tqdm(list_drawings, bar_format=bar_format, disable=silent, leave=False)

    with open(output_file, 'w') as ndjson_file:
        # Write each drawing's dict to the file as a new line
        # The json.dump is necessary to output correctly formatted JSON
        for dict_drawing in tqdm_list_drawings:
            ujson.dump(dict_drawing, ndjson_file)
            ndjson_file.write('\n')


def create_classes_mapping(class_files_path:str) -> dict:
    '''
    Create a mapping of the classes present in a directory for OHE, return it as a dictionary
    '''
    # List the class files in the directory and strip the descriptors (we filter ndjson files only)
    list_classes = []
    classes_files = [file for file in os.listdir(class_files_path) if os.path.isfile(os.path.join(class_files_path, file)) and file.endswith('.ndjson')]
    for class_file in classes_files:
        # remove the file extension
        class_name = re.sub(r'\.[^.]*$', '', class_file)
        # remove all descriptive prefixes with the class name starting after the last '_'
        class_name = re.sub(r'^.*_(.+)$', r'\1', class_name)
        list_classes.append(class_name)

    # Build a mapping dictionary of the classes
    dict_classes = {}
    for index, class_name in enumerate(list_classes) :
        dict_classes[class_name] = index

    return dict_classes



# # Download a blob from a bucket and return it as a binary I/O
# def download_blob_to_memory(bucket_name: str, source_blob_name: str) -> bytes:
#     '''
#     Downloads a blob from the bucket.
#     '''
#     # Initialize a client
#     storage_client = storage.Client()

#     # Get the bucket
#     bucket = storage_client.bucket(bucket_name)

#     # Get the blob
#     blob = bucket.blob(source_blob_name)

#     # Open the blob in binary I/O mode
#     bin_blob = blob.open(mode='rb')

#     return bin_blob



# # Download a blob from a bucket and return it as a Blob Class instance
# def download_blob_to_memory(bucket_name:str, source_blob_name:str) -> storage.Blob:
#     '''
#     Downloads a blob from the bucket.
#     '''
#     # Initialize a client
#     storage_client = storage.Client()

#     # Get the bucket
#     bucket = storage_client.bucket(bucket_name)

#     # Get the blob
#     blob = bucket.blob(source_blob_name)

#     return blob



# # Upload a blob from memory to a bucket
# def upload_blob_from_memory(source_blob: bytes, bucket_name: str, destination_blob_name: str = None) -> None:
#     '''
#     Uploads a file to the bucket.
#     '''
#     # Initialize a client
#     storage_client = storage.Client()

#     # Get the bucket
#     bucket = storage_client.bucket(bucket_name)

#     # Use the file name for the blob name if a blob name is not provided
#     if destination_blob_name is None:
#         destination_blob_name = source_blob.name

#     # Create a blob on the destination
#     target_blob = bucket.blob(destination_blob_name)

#     # Upload the file to the blob
#     target_blob.upload_from_file(source_blob.open(mode='r'))



# # Upload a blob from memory to a bucket
# def upload_blob_from_memory(source_blob:storage.Blob, bucket_name:str, destination_blob_name:str = None) -> None:
#     '''
#     Uploads a blob to the bucket.
#     '''
#     # Initialize a client
#     storage_client = storage.Client()

#     # Get the bucket
#     bucket = storage_client.bucket(bucket_name)

#     # Use the file name for the blob name if a blob name is not provided
#     if destination_blob_name is None:
#         destination_blob_name = source_blob.name

#     # Create a blob on the destination
#     target_blob = bucket.blob(destination_blob_name)

#     # Upload the file to the blob
#     target_blob.upload_from_file(source_blob.open(mode='rb'))



# The below is only here temporarily
def load_json_for_training(ndjson_filepath:object, is_X=True):
    nb_drawings_to_load = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', ndjson_filepath]))).group())
#     nb_drawings_to_load = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', ndjson_filepath]))).group())

#     for i in range(nb_drawings_to_load):
#         json_drawing = json.loads(linecache.getline(ndjson_filepath, i+1 , module_globals=None))

#         if is_X:

#             feature = json_drawing['X_list']
#             linecache.clearcache()

#             return np.array(feature)

#         else:
#             return json_drawing['Y_list']
