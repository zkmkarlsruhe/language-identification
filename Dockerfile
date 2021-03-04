FROM nvidia/cuda:11.2.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	ffmpeg libasound-dev python3 python3-pip 

COPY requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt
