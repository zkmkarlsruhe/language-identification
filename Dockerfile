#FROM nvidia/cuda:11.2.1-cudnn8-devel
FROM tensorflow/tensorflow:latest-devel-gpu

### ==== update ====
RUN apt-get update

### ==== required ====
ARG DEBIAN_FRONTEND=noninteractive

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
RUN apt-get install -y \
	ffmpeg libasound-dev python3 python3-pip 

### ==== Pip ====
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

### ==== Cuda ====
ENV PATH=/usr/local/cuda-11.2/bin:$PATH 
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
RUN export PATH=/usr/local/cuda-11.2/bin:$PATH 
RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#RUN echo $PATH
#RUN echo $LD_LIBRARY_PATH

### ==== cleanup ====
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /work/src

