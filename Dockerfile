FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install datasets
