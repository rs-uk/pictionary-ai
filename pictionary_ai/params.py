import os

#####################  CONSTANTS  #####################
MAX_LENGTH = 150 # Max number of points we keep in a drawing for training and inference

# The Google-managed buckets with the quickdraw dataset
ORIGINAL_BUCKET_DRAWINGS_SIMPLIFIED = 'quickdraw_dataset/full/simplified'
ORIGINAL_BUCKET_DRAWINGS_RAW = 'quickdraw_dataset/full/raw'


# Set LOCAL_DATA_PATH to the "raw_data" folder and create it if needed
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data")
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)


#####################  VARIABLES  #####################
BUCKET_NAME_DRAWINGS_SIMPLIFIED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED')
# BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED')
