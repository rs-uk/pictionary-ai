
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
