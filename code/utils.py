from google.cloud import storage


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
def download_blob(bucket_name, source_blob_name, destination_file_name) -> None:
    '''
    Downloads a blob from the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Download the blob to a file
    blob.download_to_filename(destination_file_name)


# Upload a file to a bucket
def upload_blob(bucket_name, source_file_name, destination_blob_name) -> None:
    '''
    Uploads a file to the bucket.
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob
    blob = bucket.blob(destination_blob_name)

    # Upload the file to the blob
    blob.upload_from_filename(source_file_name)