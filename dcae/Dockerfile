FROM everymatrix.jfrog.io/emlab-docker-remote-nvcr/nvidia/pytorch:24.12-py3

COPY dc-ae-f32c32-in-1.0 /dc-ae-f32c32-in-1.0
COPY efficientvit /efficientvit

RUN cd /efficientvit && python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

EXPOSE 6006/tcp

RUN mkdir /app
VOLUME [ "/app" ]
WORKDIR /app
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-11.html
CMD ["/bin/bash"]

