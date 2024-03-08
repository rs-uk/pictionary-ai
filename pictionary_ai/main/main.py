from pictionary_ai.model import models
from sklearn.preprocessing import OneHotEncoder
from pictionary_ai.utils import list_blobs, download_blob, upload_blob, load_json_for_training
import pandas as pd
import numpy as np
import json

#bucket name i am importing from
destination_path= '../raw_data'
destination_file_name_y= 'y_json.json'
destination_file_name_X = 'X_json.json'
source_path = ''
source_blob_name_y = 'y_json.json'
source_blob_name_X = 'X_json.json'
bucket_name = 'quickdraw-simplified-modelready'

#get blob names
blob_names = list_blobs(bucket_name=bucket_name)

#donload blod from bucket to a destination file name
# download_blob(bucket_name=bucket_name, source_blob_name=source_blob_name_y, destination_path=destination_path, destination_file_name=destination_file_name_y)
# download_blob(bucket_name=bucket_name, source_blob_name=source_blob_name_X, destination_path=destination_path, destination_file_name=destination_file_name_X)


X = load_json_for_training('../raw_data/X_json.json')
y = load_json_for_training('../raw_data/y_json.json', is_X=False)

X = np.array(X)
y = np.array(y)

X = np.array(X)
y = np.array(y)

X_y_dict = {}

for i, features in enumerate(zip(X,y)):
    
    X_y_dict[f"{i}"] = features

import random 

dict_keys = list(X_y_dict.keys())

random.shuffle(dict_keys)

shuffled_X_y = [(key, X_y_dict[key])[1] for key in dict_keys]

X_list_shuffled = []
y_list_shuffled = []

for X, y in shuffled_X_y:
    
    X_list_shuffled.append(X)
    y_list_shuffled.append(y)

X_shuffled = np.array(X_list_shuffled)
y_shuffled = np.array(y_list_shuffled)




#terget encoding, than transforming y which is the classes
#the bellow line was for getting it from cv it will nto work with buckets
target_encoder = OneHotEncoder(sparse_output=False)
#terget encoding, than transforming y which is the classes
y =target_encoder.fit_transform(y_shuffled.reshape(-1, 1))

#here XX is the X padded from processing, so need ot get it from buckets
padded_tensor = X_shuffled

tensor_length = len(padded_tensor)
train_length = int(0.7 * tensor_length)
test_length = tensor_length- train_length

#taking in the padded X data and splititng it 70 30
X_train = padded_tensor[:train_length,]
X_test = padded_tensor[train_length:,]

print(X_train)
#taking in y encoded and spliting it 70 30
y_train = y[:train_length]
y_test = y[train_length:]
print(y_train)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)


#initialize model
model = models.model_bidirectional()
#compile model
model= models.compile_model(model)
#train model
model, history = models.train_model(model, X_train, y_train, validation_data=[X_val,y_val])
#evaluate model
metrics= models.evaluate_model(model, X_test, y_test)

#upload model to bucket


upload_blob(source_path='raw_data/models',source_file_name='model_saved_pictionaryai')
