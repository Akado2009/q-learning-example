FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y ca-certificates  \
                      software-properties-common \
                      git \
                      libsm6 \
                      libxext6 \
                      libxrender-dev

# INSTALL PYTHON
RUN add-apt-repository ppa:deadsnakes/ppa && \
        apt-get install -y python3.7 python3-pip

RUN mkdir /app
ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

RUN pip3 install --upgrade tensorflow
RUN pip3 install --upgrade keras
ENTRYPOINT [ "python3.6", "main.py"]