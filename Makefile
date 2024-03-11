.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pictionary_ai || :
	@pip install -e .

run_download_simplified_dataset:
	python -c 'from pictionary_ai.interface.main import download_simplified_dataset; download_simplified_dataset()'

run_preprocess_simplified_dataset:
	python -c 'from pictionary_ai.interface.main import preprocess_simplified_dataset; preprocess_simplified_dataset()'

run_pad_preprocessed_dataset:
	python -c 'from pictionary_ai.interface.main import pad_preprocessed_dataset; pad_preprocessed_dataset()'

run_OHE_padded_dataset:
	python -c 'from pictionary_ai.interface.main import OHE_padded_dataset; OHE_padded_dataset()'

run_full_processing_dataset:
	python -c 'from pictionary_ai.interface.main import preprocess_pad_OHE_simplified_dataset; preprocess_pad_OHE_simplified_dataset()'
