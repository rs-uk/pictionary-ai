import os

#####################  CONSTANTS  #####################
MAX_LENGTH = 150 # Max number of points we keep in a drawing for training and inference

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), 'code', 'rs-uk', 'pictionary-ai', 'raw_data')




#####################  VARIABLES  #####################
BUCKET_NAME_DRAWINGS_SIMPLIFIED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED')
BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED = os.environ.get('BUCKET_NAME_DRAWINGS_SIMPLIFIED_PROCESSED')
