.DEFAULT_GOAL := help
.EXPORT_ALL_VARIABLES:

install:
	pipenv install --dev

run_eda:
	pipenv run explore_data.py

run_all:
	pipenv run explore_data.py
	pipenv run process_data.py
	pipenv run model.py

local_mlflow:
	pipenv run mlflow ui

save_docs:
	pdoc src -o docs

run_ dashboard:
	streamlit run dashboard/home_page.py
