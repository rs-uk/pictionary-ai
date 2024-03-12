# PYTHONPATH := $(PYTHONPATH)

# include pictionary_ai/params.py
# sort out the os package import in params.py so that we can use global env variables


.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pictionary_ai || :
	@pip install -e .

download_simplified_dataset:
	python -c 'from pictionary_ai.interface.main import download_simplified_dataset; download_simplified_dataset()'

run_preprocess_simplified_dataset:
	python -c 'from pictionary_ai.interface.main import preprocess_simplified_dataset; preprocess_simplified_dataset()'

run_pad_preprocessed_dataset:
	python -c 'from pictionary_ai.interface.main import pad_preprocessed_dataset; pad_preprocessed_dataset()'

run_OHE_padded_dataset:
	python -c 'from pictionary_ai.interface.main import OHE_padded_dataset; OHE_padded_dataset()'

run_full_processing_dataset:
	python -c 'from pictionary_ai.interface.main import preprocess_pad_OHE_simplified_dataset; preprocess_pad_OHE_simplified_dataset()'

run_build_subset:
	python -c 'from pictionary_ai.interface.main import generate_subset_Xy; generate_subset_Xy()'

run_build_and_split_subset:
	python -c 'from pictionary_ai.interface.main import generate_subset_Xy, split_Xy; split_Xy(generate_subset_Xy())'

run_train_model:
	python -c 'from pictionary_ai.interface.main import train_model_calling; train_model_calling()'


############### Data actions ####################
.PHONY: erase_local_processed_files
erase_local_processed_files:
	@bash erase_processed_data.sh
