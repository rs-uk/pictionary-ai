import os

#####################  VARIABLES  #####################
BUCKET_NAME_DRAWINGS_SIMPLIFIED = 'quickdraw-simplified'
# BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED')


#####################  CONSTANTS  #####################
MAX_LENGTH = 150 # Max number of points we keep in a drawing for training and inference
PADDING_VALUE = 99
NUMBER_CLASSES = 5
PERCENT_CLASS = 1

# The Google-managed buckets with the quickdraw dataset
ORIGINAL_BUCKET_DRAWINGS = 'quickdraw_dataset'
ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX = 'full/simplified'
ORIGINAL_BLOB_DRAWINGS_RAW_PREFIX = 'full/raw'


# Set LOCAL_DATA_PATH to the "raw_data" folder and create it if needed
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data")
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)

# Define the local directories based off the number of classes and ratio used in training

LOCAL_DRAWINGS_SIMPLIFIED_PATH = '/'.join((LOCAL_DATA_PATH, BUCKET_NAME_DRAWINGS_SIMPLIFIED))
LOCAL_DRAWINGS_SIMPLIFIED_PREPROCESSED_PATH = f"{LOCAL_DATA_PATH}/preprocessed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
LOCAL_DRAWINGS_SIMPLIFIED_PADDED_PATH = f"{LOCAL_DATA_PATH}/padded_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
LOCAL_DRAWINGS_SIMPLIFIED_OHE_PATH = f"{LOCAL_DATA_PATH}/OHE_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
LOCAL_DRAWINGS_SIMPLIFIED_PROCESSED_PATH = f"{LOCAL_DATA_PATH}/processed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
LOCAL_DRAWINGS_SIMPLIFIED_MODELREADY_PATH = f"{LOCAL_DATA_PATH}/model-ready_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"

# Define the local directory to save the model
MODELS_PATH = '/'.join((LOCAL_DATA_PATH, 'models'))
