# A Dockerfile that sets up a full Gymnasium install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb unzip patchelf ffmpeg cmake swig git\
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/crossworld/
WORKDIR /usr/local/crossworld/
RUN free -g
RUN pip install .[testing]
RUN git clone https://github.com/reginald-mclean/Gymnasium.git
RUN pip install -e Gymnasium


ENTRYPOINT ["/usr/local/crossworld/docker/entrypoint"]
