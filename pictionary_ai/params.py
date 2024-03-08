import os

#####################  CONSTANTS  #####################
MAX_LENGTH = 150 # Max number of points we keep in a drawing for training and inference

# Set LOCAL_DATA_PATH to the "raw_data" folder:
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")


#####################  VARIABLES  #####################
BUCKET_NAME_DRAWINGS_SIMPLIFIED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED')
BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED')
