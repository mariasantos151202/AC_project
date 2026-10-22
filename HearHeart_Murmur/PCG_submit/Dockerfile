FROM python:3.8.12-buster

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
# RUN apt install python3.8
## Include the following line if you have a requirements.txt file.
RUN pip install --upgrade pip
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN apt-get -y update
RUN apt-get install -y libsndfile1
