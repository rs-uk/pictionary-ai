from google.cloud import storage
import subprocess
import re
import linecache
import numpy as np
import json


# Copy all the content of one bucket to another bucket
def copy_bucket(source_bucket_name: str, destination_bucket_name:str) -> None:
    '''
    Copies all blobs from the source bucket to the destination bucket.
    '''
    # Initialize clients for source and destination buckets
    source_client = storage.Client()
    destination_client = storage.Client()

    # Get the source and destination buckets
    source_bucket = source_client.bucket(source_bucket_name)
    destination_bucket = destination_client.bucket(destination_bucket_name)

    # List blobs in the source bucket
    blobs = source_bucket.list_blobs()

    # Copy each blob to the destination bucket
    for blob in blobs:
        source_blob = source_bucket.blob(blob.name)
        destination_blob = destination_bucket.blob(blob.name)
        destination_blob.copy(source_blob)


# List all the blovs in a given bucket aand return their names in a list
def list_blobs(bucket_name: str) -> list:
    '''
    Lists all the blobs in the bucket.
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


# Download a blob from a bucket and store it locally
def download_blob_to_local_file(bucket_name: str, source_blob_name: str, destination_path: str, destination_file_name:str = None) -> None:
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


# Upload a local file to a bucket
def upload_blob_from_local_file(source_path: str, source_file_name:str, bucket_name: str, destination_blob_name:str = None) -> None:
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

def load_json_for_training(ndjson_filepath: object, is_X=True):

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


# Download a blob from a bucket and return it as a Blob Class instance
def download_blob_to_memory(bucket_name: str, source_blob_name: str) -> storage.Blob:
    '''
    Downloads a blob from the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    return blob


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


# Upload a blob from memory to a bucket
def upload_blob_from_memory(source_blob: storage.Blob, bucket_name: str, destination_blob_name: str = None) -> None:
    '''
    Uploads a blob to the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Use the file name for the blob name if a blob name is not provided
    if destination_blob_name is None:
        destination_blob_name = source_blob.name

    # Create a blob on the destination
    target_blob = bucket.blob(destination_blob_name)

    # Upload the file to the blob
    target_blob.upload_from_file(source_blob.open(mode='rb'))

# The below is only here temporarily
def load_json_for_training(ndjson_filepath: object, is_X=True):
    nb_drawings_to_load = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', ndjson_filepath]))).group())

    if is_X:
        with open(ndjson_filepath, 'r') as f:
            feature = json.load(f)['X_data']
        return feature
    else:
        with open(ndjson_filepath, 'r') as f:
            feature = json.load(f)['y_data']
        return feature

#     nb_drawings_to_load = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', ndjson_filepath]))).group())

#     for i in range(nb_drawings_to_load):
#         json_drawing = json.loads(linecache.getline(ndjson_filepath, i+1 , module_globals=None))

#         if is_X:

#             feature = json_drawing['X_list']
#             linecache.clearcache()

#             return np.array(feature)

#         else:
#             return json_drawing['Y_list']

