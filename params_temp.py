#MODEL (local, gcs, or now mlflow) 
MODEL_TARGET=gcs


# Set LOCAL_DATA_PATH to the "raw_data" folder directly in 1 single command:
LOCAL_RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")

# in 2 steps if above doesnt work:
# Get the absolute path of the current script/module
LOCAL_DATA_PATH = os.path.dirname(os.path.abspath(__file__))
# Create the path to the "raw_data" folder
RAW_DATA_FOLDER = os.path.join(LOCAL_DATA_PATH, "raw_data")

#Shall work too
#Current "raw_data" folder:
LOCAL_DATA_FOLDER_PATH =  os.path.abspath(os.path.join(os.getcwd(),'raw_data'))


##################  VARIABLES  ##################
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

#BUCKETS
BUCKET_NAME_DRAWINGS_SIMPLIFIED = 'quickdraw-simplified'
BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = 'quickdraw-simplified-processed'
BUCKET_NAME_MODELS= 'model_saved_pictionaryai'

#To save the model output (used in registry.py)
LOCAL_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlops","training_outputs")


