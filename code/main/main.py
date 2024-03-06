from model import models
from sklearn.preprocessing import OneHotEncoder

target_encoder = OneHotEncoder(sparse_output=False)
#terget encoding, than transforming y which is the classes
y_encoded =target_encoder.fit_transform(yy.reshape(-1,1))

#here XX is the X padded from processing
padded_tensor = XX

tensor_length = len(padded_tensor)
train_length = int(0.7 * tensor_length)
test_length = tensor_length- train_length

#taking in the padded X data and splititng it 70 30
X_train = padded_tensor[:train_length,]
X_test = padded_tensor[train_length:,]

#taking in y encoded and spliting it 70 30
y_train = y_encoded[:train_length]
y_test = y_encoded[train_length:]

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
