FROM everymatrix.jfrog.io/emlab-docker-remote-hub/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# https://hub.docker.com/r/pytorch/pytorch/tags

RUN mkdir /app

RUN apt-get update -y && apt-get install -y ffmpeg

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U xformers==0.0.29.post1; \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm==4.67.1  torch-ema==0.3 numpy==2.2.2 timm==1.0.14 tensorboard==2.18.0 einops==0.8.0 diffusers==0.32.2 matplotlib==3.10.0 ffmpeg-python==0.2.0 transformers==4.48.2



EXPOSE 6006

VOLUME [ "/app" ]
WORKDIR /app

CMD ["/bin/bash"]