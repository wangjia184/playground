FROM everymatrix.jfrog.io/emlab-docker-remote-hub/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# https://hub.docker.com/r/pytorch/pytorch/tags

RUN mkdir /app
#ADD cifar10-64.tar.gz /

RUN pip install -U xformers==0.0.29.post1;

RUN pip install tqdm==4.67.1 \
    numpy==2.2.2 \
    timm==1.0.14 \
    tensorboard==2.18.0 \
    einops==0.8.0 \
    diffusers==0.32.2 \
    matplotlib==3.10.0 \
    torchdiffeq==0.2.5 \
    einx==0.3.0 \
    hyper-connections==0.1.9 \
    ema-pytorch==0.7.7 \
    accelerate==1.3.0

EXPOSE 6006

VOLUME [ "/app" ]
WORKDIR /app

CMD ["/bin/bash"]