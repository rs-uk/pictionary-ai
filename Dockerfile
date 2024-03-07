#Insall a pockage from Docker hub
FROM python:3.10.6-buster
#We can use the following below package if we need
#FROM tensorflow/tensorflow:2.10.0
WORKDIR /prod

#install dependencies and main folder
# First, pip install dependencies
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install taxifare!
COPY code code
COPY setup.py setup.py
RUN pip install .

#Use the local caching mechanism we put in place for CSVs and Models
COPY code code
COPY Makefile Makefile
#RUN make reset_local_files

#CMD uvicorn Pictionary.api.fast:app --host 0.0.0.0
#CMD uvicorn pictionary-ai.api.fast:app --host 0.0.0.0 --port $PORT
































