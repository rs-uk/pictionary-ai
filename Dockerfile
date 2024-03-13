# declare the python env
FROM python:3.10.6-buster

# WORKDIR /prod

RUN pip install --upgrade pip

# install dependencies
COPY requirements_production.txt requirements.txt
RUN pip install -r requirements.txt

#install raw data
COPY shared_data shared_data

# install pictionary_ai package
COPY pictionary_ai /pictionary_ai
COPY setup.py setup.py
RUN pip install .

# get make functions ready
COPY Makefile Makefile
#RUN make reset_local_files

# launch the API
CMD uvicorn pictionary_ai.api.pict-api:app --host 0.0.0.0
#--port $PORT
