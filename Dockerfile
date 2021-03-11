#FROM nvidia/cuda:11.2.1-cudnn8-devel
FROM tensorflow/tensorflow:latest-devel-gpu

### ==== update ====
# RUN apt-get update
# RUN apt-get upgrade -y

### ==== required ====
#ARG DEBIAN_FRONTEND=noninteractive

### ==== SSH ==== 
#RUN apt-get install -y openssh-server
#RUN mkdir /var/run/sshd
#RUN echo 'root:root' |chpasswd
#RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
#RUN mkdir /root/.ssh
#EXPOSE 22
#CMD    ["/usr/sbin/sshd", "-D"]
### ==== END SSH ====

### ==== lid specific installs ====
#RUN apt-get install -y \
#	ffmpeg sox libasound-dev python3 python3-pip 
#RUN python3 -m pip install --upgrade pip

### ==== Pip ====
#COPY requirements.txt /requirements.txt
# enum34 did not let us install latest kapre
#RUN pip uninstall -y enum34
#RUN pip install -r requirements.txt

### ==== Cuda ====
#ENV PATH=/usr/local/cuda-11.2/bin:$PATH 
#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
#RUN export PATH=/usr/local/cuda-11.2/bin:$PATH 
#RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
#RUN export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

### ==== cleanup ====
# RUN apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# WORKDIR /work/src

