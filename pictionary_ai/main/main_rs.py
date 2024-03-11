'''
RAJ: CHANGES MADE TO HONOR'S FILE: 
Skipped most of the steps for loading the data
Loaded data from locally saved json files
Minor manipulations to get json data into format for models
Changed model variable name to model1003 and imported functions from models_rs

'''

from pictionary_ai.model import models_rs
from sklearn.preprocessing import OneHotEncoder
#from pictionary_ai.utils import list_blobs, download_blob_to_local_file, upload_blob_from_local_file, load_json_for_training
import pandas as pd
import numpy as np
import json


#bucket name i am importing from
# destination_path= '../raw_data'
# destination_file_name_y= 'y_json.json'
# destination_file_name_X = 'X_json.json'
# source_path = ''
# source_blob_name_y = 'y_json.json'
# source_blob_name_X = 'X_json.json'
# bucket_name = 'quickdraw-simplified-modelready'

# #get blob names
# blob_names = list_blobs(bucket_name=bucket_name)

# # donload blod from bucket to a destination file name

# # download_blob_to_local_file(bucket_name=bucket_name, source_blob_name=source_blob_name_y, destination_path=destination_path, destination_file_name=destination_file_name_y)
# # download_blob_to_local_file(bucket_name=bucket_name, source_blob_name=source_blob_name_X, destination_path=destination_path, destination_file_name=destination_file_name_X)

# X = load_json_for_training('../raw_data/X_json.json')
# y = load_json_for_training('../raw_data/y_json.json', is_X=False)

# X = np.array(X)
# y = np.array(y)

# # X = np.array(X)
# # y = np.array(y)

# X_y_dict = {}

# for i, features in enumerate(zip(X,y)):
    
#     X_y_dict[f"{i}"] = features

# import random 

# dict_keys = list(X_y_dict.keys())

# random.shuffle(dict_keys)

# shuffled_X_y = [(key, X_y_dict[key])[1] for key in dict_keys]

# X_list_shuffled = []
# y_list_shuffled = []

# for X, y in shuffled_X_y:
    
#     X_list_shuffled.append(X)
#     y_list_shuffled.append(y)

# X_shuffled = np.array(X_list_shuffled)
# y_shuffled = np.array(y_list_shuffled)


# #terget encoding, than transforming y which is the classes
# #the bellow line was for getting it from cv it will nto work with buckets
# target_encoder = OneHotEncoder(sparse_output=False)
# #terget encoding, than transforming y which is the classes

# y =target_encoder.fit_transform(y_shuffled.reshape(-1, 1))

# #here XX is the X padded from processing, so need ot get it from buckets
# padded_tensor = X_shuffled

# tensor_length = len(padded_tensor)
# train_length = int(0.7 * tensor_length)
# test_length = tensor_length- train_length

# #taking in the padded X data and splititng it 70 30
# X_train = padded_tensor[:train_length,]
# X_test = padded_tensor[train_length:,]

# print(X_train)
# #taking in y encoded and spliting it 70 30
# y_train = y[:train_length]
# y_test = y[train_length:]
# print(y_train)

#############################################################
###### DATA BEING PROVIDED IS ALREADY SHUFFLED ##############
###### DATA HAS ALREADY BEEN SPLIT INTO TRAIN AND TEST ######
#############################################################

#destination_path= '../raw_data'
#destination_file_name_y= 'y_json.json'
#destination_file_name_X = 'X_json.json'
source_path = '/home/jupyter/lewagon_projects/pictionary-ai/raw_data/'
file_name_X_train = 'X_train_50_classes.json'
file_name_y_train = 'y_train_50_classes.json'
file_name_X_test = 'X_test_50_classes.json'
file_name_y_test = 'y_test_50_classes.json'
#bucket_name = 'quickdraw-simplified-traintest'

# Function to load files into memory from local
def load_json_from_local(folder_path: str, file_name: str) -> list: 
    file_path = folder_path+file_name
    with open(file_path, 'r') as file : 
        json_data = json.load(file)
    print(f'Loaded {file_name} from {folder_path}')
    return json_data

# Load X_train and y_train json files
X_train_json = load_json_from_local(source_path, file_name_X_train)
y_train_json = load_json_from_local(source_path, file_name_y_train)

print('X_train and y_train loaded from json files')

# Create X_train and y_train data sets in the format required by the model
# y_train : [0] slice is needed to create y_train in correct shape (xxx, 50)
X_train = np.array([drawing['list_deltas'] for drawing in X_train_json])
y_train = np.array([drawing['OHC_class'][0] for drawing in y_train_json])  


# Split X_train & y_train in train and val sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

      
#initialize model
model1003 = models_rs.model_bidirectional()
#compile model
model1003 = models_rs.compile_model(model1003)
#train model
model1003, history = models_rs.train_model(model1003, X_train, y_train, validation_data=[X_val,y_val])

      
#evaluate model
###### LOAD IN X_test, y_test at this point in order to not overwhelm RAM ######

# Load X_test and y_test json files
X_test_json = load_json_from_local(source_path, file_name_X_test)
y_test_json = load_json_from_local(source_path, file_name_y_test)

print('X_test and y_test loaded from json files')

# Create X_test and y_test data sets in the format required by the model
# y_test : [0] slice is needed to create y_train in correct shape (xxx, 50)
X_test = np.array([drawing['list_deltas'] for drawing in X_test_json])
y_test = np.array([drawing['OHC_class'][0] for drawing in y_test_json])  

      
metrics= models_rs.evaluate_model(model1003, X_test, y_test)

# upload model to bucket

###### NOT SURE WHERE THIS FUNCTION IS DEFINED - CREATE MY OWN ######
# upload_blob(source_path='raw_data/models',source_file_name='model_saved_pictionaryai')

# upload_blob_from_local_file(
#     source_path = 'raw_data/models_1003_50classes',
#     source_file_name = 'model_saved_pictionaryai_1003_50classes',
#     bucket_name = 'model_saved_pictionaryai'
#     )    
