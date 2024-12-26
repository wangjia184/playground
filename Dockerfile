FROM everymatrix.jfrog.io/emlab-docker-remote-nvcr/nvidia/pytorch:24.12-py3

RUN mkdir /app
ADD cartoonset100k.tar /

EXPOSE 6006

VOLUME [ "/app" ]
WORKDIR /app
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-11.html
CMD ["/bin/bash"]