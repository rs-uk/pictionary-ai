from pictionary_ai.params import *
from pictionary_ai.utils import *
import numpy as np
import re, linecache, subprocess
import ujson # much faster than json lib for simple tasks
from tqdm.auto import tqdm
import random


def process_drawing_data(json_drawing:ujson) -> np.ndarray:
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


def shuffle_class(list_drawings:list) -> list:
    '''
    Shuffle drawings within a class in case there was some order in the class.
    '''
    random.shuffle(list_drawings) # works in-place

    return list_drawings


def preprocess_class(ndjson_filepath:object, nb_drawings_to_load:str, shuffle:bool = True, silent:bool = False) -> list:
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
    r_bar='| {n_fmt}/{total_fmt}'
    bar_format = l_bar + bar + r_bar
    tqdm_drawings_to_load = tqdm(range(int(nb_drawings_to_load)), bar_format=bar_format, disable=silent, leave=False)

    for i in tqdm_drawings_to_load:
        json_drawing = ujson.loads(linecache.getline(ndjson_filepath, i+1 , module_globals=None))
        np_deltas = process_drawing_data(json_drawing)
        dict_drawing = {'key_id': json_drawing['key_id'],
                        'class': json_drawing['word'],
                        'length': len(np_deltas),
                        'list_deltas': np_deltas.tolist()  # need to be transformed to list to dump as Json file later
                       }
        list_drawings.append(dict_drawing)
    tqdm_drawings_to_load.close()
    linecache.clearcache()
    # Caveat of shuffling here is we take the first nb_drawings_to_load in the class before shuffling.
    if shuffle:
        random.shuffle(list_drawings)

    return list_drawings


def add_padding(list_drawing:list, max_length:int = MAX_LENGTH) -> list:
    '''
    Apply padding to a drawing (as list) or truncate it if needed
    '''
    # Define values for padding layers - e.g. [99,99,99]
    padding = [[PADDING_VALUE,PADDING_VALUE,PADDING_VALUE]]

    # If array is greater than max_length slice off remainder of array
    if len(list_drawing) >= max_length :
        list_drawing_padded = list_drawing[0:max_length]

    # If array is less than max_length, adding padding
    else :
        pad_length = max_length - len(list_drawing)
        list_drawing_padded = list_drawing + padding * pad_length

    return list_drawing_padded


def pad_class(list_processed_drawings:list, silent:bool = False) -> list:
    '''
    Pads a list of processed drawings (as dict), returning a list of dictionaries.
    Each dictionary contains:
        - key_id, as string
        - class, as string
        - length, as integer
        - list_deltas_padded, as list
    '''

    l_bar='{percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt}'
    bar_format = l_bar + bar + r_bar
    tqdm_list_processed_drawings = tqdm(list_processed_drawings, bar_format=bar_format, disable=silent, leave=False)

    for dict_drawing in tqdm_list_processed_drawings:
        list_drawing = dict_drawing['list_deltas']
        list_drawing_padded = add_padding(list_drawing)
        dict_drawing['list_deltas'] = list_drawing_padded
    tqdm_list_processed_drawings.close()

    return list_processed_drawings


def OHE_class(list_padded_drawings:list, dict_classes_mapping:dict, silent:bool = False) -> list:
    '''
    OHE a list of padded drawings (as dict), returning a list of dictionaries.
    Each dictionary contains:
        - key_id, as string
        - class, as string
        - OHE_class, as list
        - length, as integer
        - list_deltas_padded, as list
    '''
    nb_classes = len(dict_classes_mapping)

    l_bar='{percentage:3.0f}%|'
    bar = '{bar}'
    r_bar='| {n_fmt}/{total_fmt}'
    bar_format = l_bar + bar + r_bar
    tqdm_list_padded_drawings = tqdm(list_padded_drawings, bar_format=bar_format, disable=silent, leave=False)

    # Extract the mapping only once as all drawings in the list are of the same class
    OHE_index = dict_classes_mapping[list_padded_drawings[0]['class']]

    for dict_drawing in tqdm_list_padded_drawings:
        OHE_output = [0] * nb_classes
        OHE_output[OHE_index] = 1
        dict_drawing['OHE_class'] = OHE_output
    tqdm_list_padded_drawings.close()

    return list_padded_drawings
