FROM tensorflow/tensorflow:latest-devel-gpu

### ==== update ====
RUN apt-get install --fix-missing
RUN apt-get update 
#RUN apt-get upgrade -y

### ==== lid specific installs ====
RUN apt-get install -y \
    ffmpeg sox libasound-dev python3 python3-pip 
RUN python3 -m pip install --upgrade pip

### ==== Pip ====
COPY requirements.txt /requirements.txt
# enum34 did not let us install latest kapre
RUN pip uninstall -y enum34
RUN pip install -r requirements.txt

### ==== cleanup ====
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /work/src

