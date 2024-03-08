from pictionary_ai.params import *
from pictionary_ai.utils import *
import numpy as np
import pandas as pd
import os
import json
import linecache
from google.cloud import storage
import subprocess
import re
import tqdm


def process_drawing_data(json_drawing: json) -> np.array:
    '''
    Extracts the drawing data (strokes list) from a drawing JSON file.
    Transforms the strokes from coordinates to deltas.
    Returns an np.array of deltas (d_x, d_y, end_of_stroke)
    '''
    # --- Data extraction ---
    list_strokes = json_drawing['drawing']

    x = []
    y = []
    stroke_delimiter = []
    list_points = [x, y, stroke_delimiter]

    for stroke in list_strokes:
        # Creating the third list to pass to the model with 0 all along a stroke and a 1 at the end of the stroke
        stroke_delimiter = [0.] * len(stroke[0])
        stroke_delimiter[-1] = 1
        # Concatenating x, y, and the delimiter to the new list of points
        list_points[0] += stroke[0]
        list_points[1] += stroke[1]
        list_points[2] += stroke_delimiter

    np_points = np.asarray(list_points)
    np_points = np_points.T

    # --- Processing ---
    # 1. Size normalization
    lower = np.min(np_points[:, 0:2], axis=0) # returns (x_min, y_min)
    upper = np.max(np_points[:, 0:2], axis=0) # returns (x_max, y_max)
    scale = upper - lower # returns (width, heigth)
    scale[scale == 0] = 1 # to escape a zero division for a vertical or horizontal stroke
    np_points[:, 0:2] = (np_points[:, 0:2] - lower) / scale

    # 2. Compute deltas
    np_points[1:, 0:2] -= np_points[0:-1, 0:2]
    np_points = np_points[1:, :]

    return np.round(np_points,decimals=4)


def process_class(ndjson_filepath: object, nb_drawings_to_load: str) -> list:
    '''
    Extracts drawing(s) information from a list of JSON drawings (as NDJSON),
    returning a list of dictionaries. We specify the number of drawings to load
    (in order of the NDJSON), giving a number or 'all'. Each dictionary contains:
        - key_id, as string
        - class, as string
        - length, as integer
        - list_deltas, as list
    '''
    list_drawings = []  # Initialize the list to return

    if nb_drawings_to_load == 'all':
        # Getting the number of lines in the file using a shell command (fastest way)
        nb_drawings_to_load = int(re.search(r'\d+', str(subprocess.check_output(['wc', '-l', ndjson_filepath]))).group())
    elif (isinstance(nb_drawings_to_load, str) and nb_drawings_to_load.isnumeric()) or isinstance(nb_drawings_to_load, int):
        # We also escape a number of drawings entered as an integer instead of a string
        nb_drawings_to_load = int(nb_drawings_to_load)
    else:
        nb_drawings_to_load = 0

    l_bar='{percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    bar_format = l_bar + bar + r_bar
    processing_bar = tqdm(range(int(nb_drawings_to_load)), bar_format=bar_format)

    for i in processing_bar:
        json_drawing = json.loads(linecache.getline(ndjson_filepath, i+1 , module_globals=None))
        np_deltas = process_drawing_data(json_drawing)
        dict_drawing = {'key_id': json_drawing['key_id'],
                        'class': json_drawing['word'],
                        'length': len(np_deltas),
                        'list_deltas': np_deltas.tolist()  # need to be transformed to list to dump as Json file later
                       }
        list_drawings.append(dict_drawing)
    linecache.clearcache()

    return list_drawings


def save_drawings_to_ndjson_local(list_drawings: list, output_file: str) -> None:
    '''
    Saves the drawings in the list to a local NDJSON file.
        - list_drawings: contains a dictionary for each drawing
        - output_file: the complete filepath to the target file to save/create (.ndjson)
    '''
    with open(output_file, 'w') as ndjson_file:
        # Write each drawing's dict to the file as a new line
        # The json.dump is necessary to output correctly formatted JSON
        for dict_drawing in list_drawings:
            json.dump(dict_drawing, ndjson_file)
            ndjson_file.write('\n')


def preprocess_drawings_gcs(source_bucket: str, destination_bucket: str) -> None:
    '''
    Preprocesses drawings in source_bucket and stores the resulting
    preprocessed drawings in destination_bucket.
    # TODO we ideally want to do this in-memory with no intermediray local storage.
    '''
    # Make a list of the blobs (classes) in the source bucket
    list_classes = list_blobs(source_bucket)

    # Create a tqdm bar with the relevant info and format
    l_bar='{percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {desc}'
    bar_format = l_bar + bar + r_bar
    processing_bar = tqdm(list_classes, bar_format=bar_format)

    # Process the classes (blobs) in the source bucket
    for blob_name in processing_bar:
        processing_bar.set_description("Processing %s" % blob_name)
        # Define the blob files locally
        blob_filepath = '/'.join((path_data, blob_name))
        blob_processed_filepath = '/'.join((path_data, 'test_' + blob_name))
        # Download that blob from the cloud
        # download_blob(bucket_drawings_simplified, blob_name, blob_filepath)
        # Process that blob (class)
        list_drawings = process_class(blob_filepath, 'all')
        # Save the processed drawings locally
        save_drawings_to_ndjson_local(list_drawings, blob_processed_filepath)
        # Upload the processed blobs to the cloud
        # upload_blob(bucket_drawings_simplified_processed, blob_processed_filepath, 'test_' + blob_name)
