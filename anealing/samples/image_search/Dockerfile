FROM nvcr.io/nvidia/cuda:11.0.3-devel-ubuntu18.04

ENV DEBIAN_FRONTEND NONE
WORKDIR /root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    unzip \
    pkg-config \
    git \
    wget \
    gnupg \
    curl \
    swig \
    libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip && pip3 install \
    openvino \
    openvino-dev
RUN pip3 install flask \
    flask-bootstrap4 \
    sqlalchemy \
    flask-sqlalchemy \
    sqlalchemy_utils \
    tensorflow==2.4.1 \
    pybind11 \
    numpy 

ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

RUN git clone https://github.com/yahoojapan/NGT \
    && mkdir NGT/build && cd NGT/build && cmake .. && make -j"$(nproc)" && make install \
    && cd ../python && python3 setup.py sdist \
    && pip3 install dist/ngt-*.tar.gz

# モデルの用意
## ダウンローダーのインストール
RUN git clone https://github.com/openvinotoolkit/open_model_zoo.git \
    && rm -r open_model_zoo/.git open_model_zoo/models/intel/*/description open_model_zoo/demos \
    && cd open_model_zoo/tools/downloader/  && pip3 install -r requirements.in
# 画像検索モデルのダウンロード
RUN python3 /root/open_model_zoo/tools/downloader/downloader.py --name image-retrieval-0001 --precisions FP16-INT8 --o models


# もう一つモデルを用意
COPY scripts/get_keras_model.py /opt/scripts/
RUN pip3 install numpy==1.18
RUN mkdir -p /root/models/effnet \
    && cd /root/models/effnet \
    && python3 /opt/scripts/get_keras_model.py effnet \
    && python3 /usr/local/lib/python3.6/dist-packages/mo_tf.py  --saved_model_dir effnet -b 1 --data_type half

RUN pip3 install dwave-ocean-sdk
RUN pip3 uninstall cmake && pip3 install cmake
RUN pip3 install openjij --no-binary :all: --no-cache-dir

