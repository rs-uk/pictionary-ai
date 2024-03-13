import os, ujson, linecache

#####################  VARIABLES  #####################
# BUCKET_NAME_DRAWINGS_SIMPLIFIED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED')
# BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED')


#####################  CONSTANTS  #####################

# Shared data path
LOCAL_SHARED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared_data")
if not os.path.exists(LOCAL_SHARED_DATA_PATH):
    os.makedirs(LOCAL_SHARED_DATA_PATH)


MAX_LENGTH = 150 # Max number of points we keep in a drawing for training and inference
PADDING_VALUE = 99
NUMBER_CLASSES = 50
PERCENT_CLASS = 25
DICT_OHE = ujson.loads(linecache.getline(f"{LOCAL_SHARED_DATA_PATH}/dict_50_class_subset.json", 1, module_globals=None))

##### /!\ if this is not None then we use the classes set above in the whole program /!\ #####
LIST_CLASSES = list(DICT_OHE.keys())

# The Google-managed buckets with the quickdraw dataset
ORIGINAL_BUCKET_DRAWINGS = 'quickdraw_dataset'
ORIGINAL_BLOB_DRAWINGS_SIMPLIFIED_PREFIX = 'full/simplified'
ORIGINAL_BLOB_DRAWINGS_RAW_PREFIX = 'full/raw'

# Our buckets with the quickdraw dataset
BUCKET_NAME_DRAWINGS_SIMPLIFIED = 'quickdraw-simplified'

# Our intermediary buckets
# BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = 'quickdraw-simplified-processed'


# Set LOCAL_DATA_PATH to the "raw_data" folder and create it if needed
### to be renamed to 'data' later on
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data")
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)

LOCAL_RAW_DATA_PATH = os.path.join(LOCAL_DATA_PATH, 'raw_data')
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)

LOCAL_PROCESSED_DATA_PATH = os.path.join(LOCAL_DATA_PATH, 'processed_data')
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)

# to rename LOCAL_MODELS_DATA_PATH
MODELS_PATH = os.path.join(LOCAL_DATA_PATH, 'models')
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs(LOCAL_DATA_PATH)


##### Define the local directories based off the number of classes and ratio used in training
# all the original quickdraw simplified drawings (346 classes)
LOCAL_DRAWINGS_SIMPLIFIED_PATH = os.path.join(LOCAL_RAW_DATA_PATH, BUCKET_NAME_DRAWINGS_SIMPLIFIED)
# original quickdraw simplified drawings used in the subset
LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# preprocessed drawings from the subset (not used when doing the processing in one step)
LOCAL_DRAWINGS_SIMPLIFIED_PREPROCESSED_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/preprocessed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# preprocessed and padded drawings from the subset (not used when doing the processing in one step)
LOCAL_DRAWINGS_SIMPLIFIED_PADDED_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/padded_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# preprocessed, padded and OHE drawings from the subset (not used when doing the processing in one step)
LOCAL_DRAWINGS_SIMPLIFIED_OHE_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/OHE_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# fully processed drawings from the subset
LOCAL_DRAWINGS_SIMPLIFIED_PROCESSED_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/processed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# model-ready Xs and ys from the subset
LOCAL_DRAWINGS_SIMPLIFIED_MODELREADY_PATH = f"{LOCAL_PROCESSED_DATA_PATH}/model-ready_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"


# ##### Define the local directories based off the number of classes and ratio used in training
# # all the original quickdraw simplified drawings (346 classes)
# LOCAL_DRAWINGS_SIMPLIFIED_PATH = '/'.join((LOCAL_DATA_PATH, BUCKET_NAME_DRAWINGS_SIMPLIFIED))
# # original quickdraw simplified drawings used in the subset
# LOCAL_DRAWINGS_SIMPLIFIED_SUBSET_PATH = f"{LOCAL_DATA_PATH}/{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# # preprocessed drawings from the subset (not used when doing the processing in one step)
# LOCAL_DRAWINGS_SIMPLIFIED_PREPROCESSED_PATH = f"{LOCAL_DATA_PATH}/preprocessed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# # preprocessed and padded drawings from the subset (not used when doing the processing in one step)
# LOCAL_DRAWINGS_SIMPLIFIED_PADDED_PATH = f"{LOCAL_DATA_PATH}/padded_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# # preprocessed, padded and OHE drawings from the subset (not used when doing the processing in one step)
# LOCAL_DRAWINGS_SIMPLIFIED_OHE_PATH = f"{LOCAL_DATA_PATH}/OHE_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# # fully processed drawings from the subset
# LOCAL_DRAWINGS_SIMPLIFIED_PROCESSED_PATH = f"{LOCAL_DATA_PATH}/processed_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"
# # model-ready Xs and ys from the subset
# LOCAL_DRAWINGS_SIMPLIFIED_MODELREADY_PATH = f"{LOCAL_DATA_PATH}/model-ready_{BUCKET_NAME_DRAWINGS_SIMPLIFIED}_{NUMBER_CLASSES}classes_{PERCENT_CLASS}pc"


# # Define the local directory to save the model
# MODELS_PATH = '/'.join((LOCAL_DATA_PATH, 'models/models'))
