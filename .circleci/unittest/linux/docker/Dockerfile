FROM ubuntu:18.04 as builder

RUN apt update -q

################################################################################
# Build Kaldi
################################################################################
RUN apt install -q -y \
        autoconf \
        automake \
        bzip2 \
        g++ \
        gfortran \
        git \
        libatlas-base-dev \
        libtool \
        make \
        python2.7 \
        python3 \
        sox \
        subversion \
        unzip \
        wget \
        zlib1g-dev

# KALDI uses MKL as a default math library, but we are going to copy featbin binaries and dependent
# shared libraries to the final image, so we use ATLAS, which is easy to reinstall in the final image.
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
        cd /opt/kaldi/tools && \
        make -j $(nproc) && \
        cd /opt/kaldi/src && \
        ./configure --shared --mathlib=ATLAS --use-cuda=no && \
        make featbin -j $(nproc)

# Copy featbins and dependent libraries
ADD ./scripts /scripts
RUN bash /scripts/copy_kaldi_executables.sh /opt/kaldi /kaldi

################################################################################
# Build the final image
################################################################################
FROM BASE_IMAGE
RUN apt update && apt install -y \
        g++ \
        gfortran \
        git \
        libatlas3-base \
        libsndfile1 \
        wget \
        curl \
        make \
        file \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /kaldi /kaldi
ENV PATH="${PATH}:/kaldi/bin" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/kaldi/lib"
